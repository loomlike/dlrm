# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from typing import Any, Dict, Iterable
import warnings

import numpy as np
import sklearn.metrics
import torch
import torch.nn as nn
from torch._ops import ops
from torch.nn.parallel.parallel_apply import parallel_apply
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.scatter_gather import gather, scatter
from torch.nn.parameter import Parameter
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard import SummaryWriter

# Local packages
from tricks.md_embedding_bag import PrEmbeddingBag, md_solver  # mixed-dimension trick
from tricks.qr_embedding_bag import QREmbeddingBag  # quotient-remainder trick

import dlrm_data_pytorch as dp  # TODO replace this to generic data loader
from dlrm_s_pytorch import (
    dlrm_wrap,
    loss_fn_wrap,
    time_wrap,
    unpack_batch,
    DLRM_Net,
    LRPolicyScheduler,
)
import extend_distributed as ext_dist
import optim.rwsadagrad as RowWiseSparseAdagrad


def run():

    # distribute data parallel mlps
    if ext_dist.my_size > 1:
        if use_gpu:
            device_ids = [ext_dist.my_local_rank]
            dlrm.bot_l = ext_dist.DDP(dlrm.bot_l, device_ids=device_ids)
            dlrm.top_l = ext_dist.DDP(dlrm.top_l, device_ids=device_ids)
        else:
            dlrm.bot_l = ext_dist.DDP(dlrm.bot_l)
            dlrm.top_l = ext_dist.DDP(dlrm.top_l)

    if not args.inference_only:
        if use_gpu and args.optimizer in ["rwsadagrad", "adagrad"]:
            sys.exit("GPU version of Adagrad is not supported by PyTorch.")
        # specify the optimizer algorithm
        opts = {
            "sgd": torch.optim.SGD,
            "rwsadagrad": RowWiseSparseAdagrad.RWSAdagrad,
            "adagrad": torch.optim.Adagrad,
        }

        parameters = (
            dlrm.parameters()
            if ext_dist.my_size == 1
            else [
                {
                    "params": [p for emb in dlrm.emb_l for p in emb.parameters()],
                    "lr": args.learning_rate,
                },
                # TODO check this lr setup
                # bottom mlp has no data parallelism
                # need to check how do we deal with top mlp
                {
                    "params": dlrm.bot_l.parameters(),
                    "lr": args.learning_rate,
                },
                {
                    "params": dlrm.top_l.parameters(),
                    "lr": args.learning_rate,
                },
            ]
        )
        optimizer = opts[args.optimizer](parameters, lr=args.learning_rate)
        lr_scheduler = LRPolicyScheduler(
            optimizer,
            args.lr_num_warmup_steps,
            args.lr_decay_start_step,
            args.lr_num_decay_steps,
        )

    ### main loop ###

    # training or inference
    best_acc_test = 0
    best_auc_test = 0
    skip_upto_epoch = 0
    skip_upto_batch = 0
    total_time = 0
    total_loss = 0
    total_iter = 0
    total_samp = 0

    # Load model is specified
    if not (args.load_model == ""):
        print("Loading saved model {}".format(args.load_model))
        if use_gpu:
            if dlrm.ndevices > 1:
                # NOTE: when targeting inference on multiple GPUs,
                # load the model as is on CPU or GPU, with the move
                # to multiple GPUs to be done in parallel_forward
                ld_model = torch.load(args.load_model)
            else:
                # NOTE: when targeting inference on single GPU,
                # note that the call to .to(device) has already happened
                ld_model = torch.load(
                    args.load_model,
                    map_location=torch.device("cuda")
                    # map_location=lambda storage, loc: storage.cuda(0)
                )
        else:
            # when targeting inference on CPU
            ld_model = torch.load(args.load_model, map_location=torch.device("cpu"))
        dlrm.load_state_dict(ld_model["state_dict"])
        ld_j = ld_model["iter"]
        ld_k = ld_model["epoch"]
        ld_nepochs = ld_model["nepochs"]
        ld_nbatches = ld_model["nbatches"]
        ld_nbatches_test = ld_model["nbatches_test"]
        ld_train_loss = ld_model["train_loss"]
        ld_total_loss = ld_model["total_loss"]
        ld_acc_test = ld_model["test_acc"]
        if not args.inference_only:
            optimizer.load_state_dict(ld_model["opt_state_dict"])
            best_acc_test = ld_acc_test
            total_loss = ld_total_loss
            skip_upto_epoch = ld_k  # epochs
            skip_upto_batch = ld_j  # batches
        else:
            args.print_freq = ld_nbatches
            args.test_freq = 0

        print(
            "Saved at: epoch = {:d}/{:d}, batch = {:d}/{:d}, ntbatch = {:d}".format(
                ld_k, ld_nepochs, ld_j, ld_nbatches, ld_nbatches_test
            )
        )
        print(
            "Training state: loss = {:.6f}".format(
                ld_train_loss,
            )
        )
        print("Testing state: accuracy = {:3.3f} %".format(ld_acc_test * 100))

    if args.inference_only:
        # Currently only dynamic quantization with INT8 and FP16 weights are
        # supported for MLPs and INT4 and INT8 weights for EmbeddingBag
        # post-training quantization during the inference.
        # By default we don't do the quantization: quantize_{mlp,emb}_with_bit == 32 (FP32)
        assert args.quantize_mlp_with_bit in [
            8,
            16,
            32,
        ], "only support 8/16/32-bit but got {}".format(args.quantize_mlp_with_bit)
        assert args.quantize_emb_with_bit in [
            4,
            8,
            32,
        ], "only support 4/8/32-bit but got {}".format(args.quantize_emb_with_bit)
        if args.quantize_mlp_with_bit != 32:
            if args.quantize_mlp_with_bit in [8]:
                quantize_dtype = torch.qint8
            else:
                quantize_dtype = torch.float16
            dlrm = torch.quantization.quantize_dynamic(
                dlrm, {torch.nn.Linear}, quantize_dtype
            )
        if args.quantize_emb_with_bit != 32:
            dlrm.quantize_embedding(args.quantize_emb_with_bit)
            # print(dlrm)
        assert (
            args.data_generation == "dataset"
        ), "currently only dataset loader provides testset"

    print("time/loss/accuracy (if enabled):")

    tb_file = "./" + args.tensor_board_filename
    writer = SummaryWriter(tb_file)

    ext_dist.barrier()
    if not args.inference_only:
        k = 0
        total_time_begin = 0
        while k < args.nepochs:
            if k < skip_upto_epoch:
                continue

            for j, input_batch in enumerate(train_ld):
                if j < skip_upto_batch:
                    continue

                X, lS_o, lS_i, T, W, CBPP = unpack_batch(input_batch)

                t1 = time_wrap(use_gpu)

                # early exit if nbatches was set by the user and has been exceeded
                if nbatches > 0 and j >= nbatches:
                    break

                # Skip the batch if batch size not multiple of total ranks
                if ext_dist.my_size > 1 and X.size(0) % ext_dist.my_size != 0:
                    print(
                        "Warning: Skiping the batch %d with size %d"
                        % (j, X.size(0))
                    )
                    continue

                mbs = T.shape[0]  # = args.mini_batch_size except maybe for last

                # forward pass
                Z = dlrm_wrap(
                    X,
                    lS_o,
                    lS_i,
                    use_gpu,
                    device,
                    ndevices=ndevices,
                )

                if ext_dist.my_size > 1:
                    T = T[ext_dist.get_my_slice(mbs)]
                    W = W[ext_dist.get_my_slice(mbs)]

                # loss
                E = loss_fn_wrap(Z, T, use_gpu, device)

                # compute loss and accuracy
                L = E.detach().cpu().numpy()  # numpy array
                # training accuracy is not disabled
                # S = Z.detach().cpu().numpy()  # numpy array
                # T = T.detach().cpu().numpy()  # numpy array

                # # print("res: ", S)

                # # print("j, train: BCE, shifted_BCE ", j, L, L_shifted)

                # mbs = T.shape[0]  # = args.mini_batch_size except maybe for last
                # A = np.sum((np.round(S, 0) == T).astype(np.uint8))
                # A_shifted = np.sum((np.round(S_shifted, 0) == T).astype(np.uint8))

                # DLRM backward
                # scaled error gradient propagation
                # (where we do not accumulate gradients across mini-batches)

                # If need gradient accumulation, control zero_grad() and step()

                optimizer.zero_grad()
                # backward pass
                E.backward()

                # optimizer
                optimizer.step()
                lr_scheduler.step()

                t2 = time_wrap(use_gpu)
                total_time += t2 - t1

                total_loss += L * mbs
                total_iter += 1
                total_samp += mbs

                should_print = ((j + 1) % args.print_freq == 0) or (
                    j + 1 == nbatches
                )
                should_test = (
                    (args.test_freq > 0)
                    and (args.data_generation == "dataset")
                    and (((j + 1) % args.test_freq == 0) or (j + 1 == nbatches))
                )

                # print time, loss and accuracy
                if should_print or should_test:
                    gT = 1000.0 * total_time / total_iter if args.print_time else -1
                    total_time = 0

                    train_loss = total_loss / total_samp
                    total_loss = 0

                    str_run_type = (
                        "inference" if args.inference_only else "training"
                    )

                    wall_time = ""

                    print(
                        "Finished {} it {}/{} of epoch {}, {:.2f} ms/it,".format(
                            str_run_type, j + 1, nbatches, k, gT
                        )
                        + " loss {:.6f}".format(train_loss)
                        + wall_time,
                        flush=True,
                    )

                    log_iter = nbatches * k + j + 1
                    writer.add_scalar("Train/Loss", train_loss, log_iter)

                    total_iter = 0
                    total_samp = 0

                # testing
                if should_test:
                    epoch_num_float = (j + 1) / len(train_ld) + k + 1

                    print(
                        "Testing at - {}/{} of epoch {},".format(j + 1, nbatches, k)
                    )
                    # TODO change to refactor
                    model_metrics_dict, is_best = inference(
                        args,
                        dlrm,
                        best_acc_test,
                        best_auc_test,
                        test_ld,
                        device,
                        use_gpu,
                        log_iter,
                    )

                    if (
                        is_best
                        and not (args.save_model == "")
                        and not args.inference_only
                    ):
                        model_metrics_dict["epoch"] = k
                        model_metrics_dict["iter"] = j + 1
                        model_metrics_dict["train_loss"] = train_loss
                        model_metrics_dict[
                            "opt_state_dict"
                        ] = optimizer.state_dict()
                        print("Saving model to {}".format(args.save_model))
                        torch.save(model_metrics_dict, args.save_model)

                    # Uncomment the line below to print out the total time with overhead
                    # print("Total test time for this group: {}" \
                    # .format(time_wrap(use_gpu) - accum_test_time_begin))
            k += 1  # nepochs
    else:
        print("Testing for inference only")
        inference(
            args,
            dlrm,
            best_acc_test,
            best_auc_test,
            test_ld,
            device,
            use_gpu,
        )

    # plot compute graph
    if args.plot_compute_graph:
        sys.exit(
            "ERROR: Please install pytorchviz package in order to use the"
            + " visualization. Then, uncomment its import above as well as"
            + " three lines below and run the code again."
        )
        # V = Z.mean() if args.inference_only else E
        # dot = make_dot(V, params=dict(dlrm.named_parameters()))
        # dot.render('dlrm_s_pytorch_graph') # write .pdf file

    # test prints
    if not args.inference_only and args.debug_mode:
        print("updated parameters (weights and bias):")
        for param in dlrm.parameters():
            print(param.detach().cpu().numpy())

    total_time_end = time_wrap(use_gpu)


# TODO
class DLRM:
    # TODO maybe use kwargs and pass to DLRM_NET if these params are too much? 
    def __init__(
        self,

        use_gpu: bool = False,

        # distributed processing
        dist_local_rank: int = -1,
        dist_backend: str = "",

        # Model architecture
        arch_dense_feature_size: int = 4,
        arch_sparse_feature_size: int = 2,
        arch_mlp_bot_hiddens: Iterable[int] = (3,),  # Bottom MLP becomes [dense_feature_size, hiddens, sparse_feature_size or 2x]
        arch_mlp_top: Iterable[int] = (4, 2, 1),
        arch_embedding_table_sizes: Iterable[int] = (3, 3),  # Number of categories for each sparse features
        arch_interaction_op: str = 'dot',  # choices=["dot", "cat"]
        arch_interaction_itself: bool = False,

        # Quotient-Remainder parameters
        qr_operation: str = None,  # if None, don't use. {'mult', 'concat'} TODO anything else???
        qr_threshold: int = 200,
        qr_collisions: int = 4,
        qr_flag: bool = False,

        # embedding table options
        md_threshold: int = 200,
        md_temperature: float = 0.3,
        md_round_dims: bool = False,
        md_flag: bool = False,

        weighted_pooling: str = None,  # either "fixed" or "learned"

        # quantize
        quantize_mlp_with_bit: int = 32,
        quantize_emb_with_bit: int = 32,

        sync_dense_params: bool = True,
        loss_threshold: float = 0.0, 

        loss_function: str = 'mse',  # or 'bce' or 'wbce'
        loss_weights: Iterable[float] = (1.0, 1.0,),  # for wbce

        # random seed
        seed: int = None,
    ) -> None:
        """

        Args:
            use_gpu: Use GPU or not

            dist_local_rank:
            dist_backend:

            seed: Random seed

        # model related parameters

        # j will be replaced with the table number


        # activations and loss
        parser.add_argument("--activation-function", type=str, default="relu")

        parser.add_argument("--round-targets", type=bool, default=False)
        # data
        parser.add_argument("--data-size", type=int, default=1)
        parser.add_argument("--num-batches", type=int, default=0)
        parser.add_argument(
            "--data-generation", type=str, default="random"
        )  # synthetic or dataset
        parser.add_argument(
            "--rand-data-dist", type=str, default="uniform"
        )  # uniform or gaussian
        parser.add_argument("--rand-data-min", type=float, default=0)
        parser.add_argument("--rand-data-max", type=float, default=1)
        parser.add_argument("--rand-data-mu", type=float, default=-1)
        parser.add_argument("--rand-data-sigma", type=float, default=1)
        parser.add_argument("--data-trace-file", type=str, default="./input/dist_emb_j.log")
        parser.add_argument("--data-set", type=str, default="kaggle")  # or terabyte
        parser.add_argument("--raw-data-file", type=str, default="")
        parser.add_argument("--processed-data-file", type=str, default="")
        parser.add_argument("--data-randomize", type=str, default="total")  # or day or none
        parser.add_argument("--data-trace-enable-padding", type=bool, default=False)
        parser.add_argument("--data-sub-sample-rate", type=float, default=0.0)  # in [0, 1]
        parser.add_argument("--num-indices-per-lookup", type=int, default=10)
        parser.add_argument("--num-indices-per-lookup-fixed", type=bool, default=False)
        parser.add_argument("--num-workers", type=int, default=0)
        parser.add_argument("--memory-map", action="store_true", default=False)
        # training
        parser.add_argument("--mini-batch-size", type=int, default=1)
        parser.add_argument("--nepochs", type=int, default=1)
        parser.add_argument("--learning-rate", type=float, default=0.01)
        parser.add_argument("--optimizer", type=str, default="sgd")
        parser.add_argument(
            "--dataset-multiprocessing",
            action="store_true",
            default=False,
            help="The Kaggle dataset can be multiprocessed in an environment \
                            with more than 7 CPU cores and more than 20 GB of memory. \n \
                            The Terabyte dataset can be multiprocessed in an environment \
                            with more than 24 CPU cores and at least 1 TB of memory.",
        )
        # inference
        parser.add_argument("--inference-only", action="store_true", default=False)


        # debugging and profiling
        parser.add_argument("--print-freq", type=int, default=1)
        parser.add_argument("--test-freq", type=int, default=-1)
        parser.add_argument("--print-time", action="store_true", default=False)
        parser.add_argument("--plot-compute-graph", action="store_true", default=False)
        parser.add_argument("--tensor-board-filename", type=str, default="run_kaggle_pt")
        # store/load model
        parser.add_argument("--save-model", type=str, default="")
        parser.add_argument("--load-model", type=str, default="")
        # LR policy
        parser.add_argument("--lr-num-warmup-steps", type=int, default=0)
        parser.add_argument("--lr-decay-start-step", type=int, default=0)
        parser.add_argument("--lr-num-decay-steps", type=int, default=0)
        

        """
        if weighted_pooling is not in {None, 'fixed', 'learned'}:
            raise ValueError("Weighted pooling should be either None, 'fixed' or 'learned'")
        if weighted_pooling is not None:
            if qr_flag:
                raise ValueError("Quotient remainder with weighted pooling is not supported")
            if md_flag:
                raise ValueError("Mixed dimensions with weighted pooling is not supported")
        if quantize_emb_with_bit in {4, 8}:
            if qr_flag:
                raise ValueError("4 and 8-bit quantization with quotient remainder is not supported")
            if md_flag:
                raise ValueError("ERROR: 4 and 8-bit quantization with mixed dimensions is not supported")

        np.random.seed(seed)
        torch.manual_seed(seed)

        ##### Device setup
        self.use_gpu = use_gpu
        if self.use_gpu and not torch.cuda.is_available():
            warnings.warn("CUDA is not available, but configured to use GPU.")
            self.use_gpu = False

        if self.use_gpu:
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            if ext_dist.my_size > 1:
                self.ndevices = 1
                self.device = torch.device("cuda", ext_dist.my_local_rank)
            else:
                self.ndevices = torch.cuda.device_count()
                self.device = torch.device("cuda", 0)
            print(f"Using {self.ndevices} GPU(s)...")
        else:
            self.ndevices = -1
            self.device = torch.device("cpu")
            print("Using CPU...")

        ext_dist.init_distributed(
            local_rank=dist_local_rank,
            use_gpu=self.use_gpu,
            backend=dist_backend,
        )

        ##### Model architecture
        # Bottom MLP
        self.m_den = arch_dense_feature_size
        self.m_spa = arch_sparse_feature_size  # Sparse feature embedding output size)
        if qr_flag:
        if qr_operation == "concat":
            # The last dim of bottom mlp must match 2x the embedding dim
            self.ln_bot = np.array([self.m_den] + list(arch_mlp_bot_hiddens) + [2 * self.m_spa])
        else:
            # The last dim of bottom mlp must match the embedding dim
            self.ln_bot = np.array([self.m_den] + list(arch_mlp_bot_hiddens) + [self.m_spa])

        # Embedding tables for sparse features
        self.ln_emb = np.array(arch_embedding_table_sizes)

        # Number of features = number of sparse features + 1 dense feature
        self.num_fea = len(self.ln_emb) + 1

        # Assign mixed dimensions if applicable
        if md_flag:
            self.m_spa = md_solver(
                torch.tensor(self.ln_emb),
                md_temperature,  # alpha
                d0=self.m_spa,
                round_dim=md_round_dims,
            ).tolist()

        # Top MLP
        m_den_out = self.ln_bot[-1]
        if arch_interaction_op == "dot":
            if arch_interaction_itself:
                num_int = (self.num_fea * (self.num_fea + 1)) // 2 + m_den_out
            else:
                num_int = (self.num_fea * (self.num_fea - 1)) // 2 + m_den_out
        elif arch_interaction_op == "cat":
            num_int = self.num_fea * m_den_out
        else:
            raise ValueError(f"ERROR: unknown arch_interaction_op: {arch_interaction_op}")
        self.ln_top = np.array([num_int] + list(arch_mlp_top))

        # Construct the neural network. Use the original implementation of FBR
        # except args.loss_function and args.loss_weight
        self.dlrm = DLRM_Net(
            m_spa=self.m_spa,
            ln_emb=self.ln_emb,
            ln_bot=self.ln_bot,
            ln_top=self.ln_top,
            arch_interaction_op=arch_interaction_op,
            arch_interaction_itself=arch_interaction_itself,
            sigmoid_bot=-1,
            sigmoid_top=self.ln_top.size-2,
            sync_dense_params=sync_dense_params,
            loss_threshold=loss_threshold,
            ndevices=self.ndevices,
            qr_flag=qr_flag,
            qr_operation=qr_operation,
            qr_collisions=qr_collisions,
            qr_threshold=qr_threshold,
            md_flag=md_flag,
            md_threshold=md_threshold,
            weighted_pooling=weighted_pooling,
        )

        if self.use_gpu:
            # Custom Model-Data Parallel
            # the mlps are replicated and use data parallelism, while
            # the embeddings are distributed and use model parallelism
            self.dlrm = self.dlrm.to(self.device)
            if self.ndevices > 1:
                self.dlrm.emb_l, self.dlrm.v_W_l = self.dlrm.create_emb(
                    self.m_spa, self.ln_emb, weighted_pooling
                )
            else:
                if weighted_pooling == 'fixed':
                    for k, w in enumerate(self.dlrm.v_W_l):
                        self.dlrm.v_W_l[k] = w.cuda()

    def fit(
        self,
        tr,
        num_batches: int = 0,
        mini_batch_size: int,
    ) -> None:
        nbatches = num_batches if num_batches > 0 else len(tr)  # TODO was len(train_ld)
        pass

    def predict(
        self,
        te,
        mini_batch_size: int = -1,
        num_workers: int = -1,
    ) -> Any:

        if mini_batch_size < 0:
            mini_batch_size = self.mini_batch_size
        if num_workers < 0:
            num_workers = self.num_workers

        # TODO...
    # def inference(
    #     args,
    #     dlrm,
    #     best_acc_test,
    #     best_auc_test,
    #     test_ld,
    #     device,
    #     use_gpu,
    #     log_iter=-1,
    # ):
        test_accu = 0
        test_samp = 0

        for i, testBatch in enumerate(test_ld):
            # early exit if nbatches was set by the user and was exceeded
            if nbatches > 0 and i >= nbatches:
                break

            X_test, lS_o_test, lS_i_test, T_test, W_test, CBPP_test = unpack_batch(
                testBatch
            )

            # Skip the batch if batch size not multiple of total ranks
            if ext_dist.my_size > 1 and X_test.size(0) % ext_dist.my_size != 0:
                print("Warning: Skiping the batch %d with size %d" % (i, X_test.size(0)))
                continue

            # forward pass
            Z_test = dlrm_wrap(
                X_test,
                lS_o_test,
                lS_i_test,
                use_gpu,
                device,
                ndevices=ndevices,
            )
            ### gather the distributed results on each rank ###
            # For some reason it requires explicit sync before all_gather call if
            # tensor is on GPU memory
            if Z_test.is_cuda:
                torch.cuda.synchronize()
            (_, batch_split_lengths) = ext_dist.get_split_lengths(X_test.size(0))
            if ext_dist.my_size > 1:
                Z_test = ext_dist.all_gather(Z_test, batch_split_lengths)

            # DLRM accuracy compute
            S_test = Z_test.detach().cpu().numpy()  # numpy array
            T_test = T_test.detach().cpu().numpy()  # numpy array

            mbs_test = T_test.shape[0]  # = mini_batch_size except last
            A_test = np.sum((np.round(S_test, 0) == T_test).astype(np.uint8))

            test_accu += A_test
            test_samp += mbs_test

        # metrics = {
        #     "recall": lambda y_true, y_score: sklearn.metrics.recall_score(
        #         y_true=y_true, y_pred=np.round(y_score)
        #     ),
        #     "precision": lambda y_true, y_score: sklearn.metrics.precision_score(
        #         y_true=y_true, y_pred=np.round(y_score)
        #     ),
        #     "f1": lambda y_true, y_score: sklearn.metrics.f1_score(
        #         y_true=y_true, y_pred=np.round(y_score)
        #     ),
        #     "ap": sklearn.metrics.average_precision_score,
        #     "roc_auc": sklearn.metrics.roc_auc_score,
        #     "accuracy": lambda y_true, y_score: sklearn.metrics.accuracy_score(
        #         y_true=y_true, y_pred=np.round(y_score)
        #     ),
        # }
        acc_test = test_accu / test_samp
        writer.add_scalar("Test/Acc", acc_test, log_iter)

        model_metrics_dict = {
            "nepochs": args.nepochs,
            "nbatches": nbatches,
            "nbatches_test": nbatches_test,  # <-- len(test_ld)
            "state_dict": dlrm.state_dict(),
            "test_acc": acc_test,
        }

        # is_best = validation_results["roc_auc"] > best_auc_test
        # if is_best:
        #     best_auc_test = validation_results["roc_auc"]
        #     model_metrics_dict["test_auc"] = best_auc_test
        # print(
        #     "recall {:.4f}, precision {:.4f},".format(
        #         validation_results["recall"],
        #         validation_results["precision"],
        #     )
        #     + " f1 {:.4f}, ap {:.4f},".format(
        #         validation_results["f1"], validation_results["ap"]
        #     )
        #     + " auc {:.4f}, best auc {:.4f},".format(
        #         validation_results["roc_auc"], best_auc_test
        #     )
        #     + " accuracy {:3.3f} %, best accuracy {:3.3f} %".format(
        #         validation_results["accuracy"] * 100, best_acc_test * 100
        #     ),
        #     flush=True,
        # )
        is_best = acc_test > best_acc_test
        if is_best:
            best_acc_test = acc_test
        print(
            " accuracy {:3.3f} %, best {:3.3f} %".format(
                acc_test * 100, best_acc_test * 100
            ),
            flush=True,
        )

        return model_metrics_dict, is_best








    def save(self):
        pass

    def load(self, path: str):
        pass

    def summary(self):
        """Print model architecture"""
        print("model arch:")
        print(
            "mlp top arch "
            + str(self.ln_top.size - 1)
            + " layers, with input to output dimensions:"
        )
        print(self.ln_top)
        print("# of interactions")
        print(self.num_int)
        print(
            "mlp bot arch "
            + str(self.ln_bot.size - 1)
            + " layers, with input to output dimensions:"
        )
        print(self.ln_bot)
        print("# of features (sparse and dense)")
        print(self.num_fea)
        print("dense feature size")
        print(self.m_den)
        print("sparse feature size")
        print(self.m_spa)
        print(
            "# of embeddings (= # of sparse features) "
            + str(self.ln_emb.size)
            + ", with dimensions "
            + str(self.m_spa)
            + "x:"
        )
        print(self.ln_emb)

        print("data (inputs and targets):")
        for j, input_batch in enumerate(self.train_ld):
            X, lS_o, lS_i, T, W, CBPP = unpack_batch(input_batch)

            torch.set_printoptions(precision=4)
            # early exit if nbatches was set by the user and has been exceeded
            if self.nbatches > 0 and j >= self.nbatches:
                break
            print("mini-batch: %d" % j)
            print(X.detach().cpu())
            # transform offsets to lengths when printing
            print(
                torch.IntTensor(
                    [
                        np.diff(
                            S_o.detach().cpu().tolist() + list(lS_i[i].shape)
                        ).tolist()
                        for i, S_o in enumerate(lS_o)
                    ]
                )
            )
            print([S_i.detach().cpu() for S_i in lS_i])
            print(T.detach().cpu())

    def summary_params(self):
        """Print model parameters"""
        print("Parameters (weights and bias):")
        for param in self.dlrm.parameters():
            print(param.detach().cpu().numpy())

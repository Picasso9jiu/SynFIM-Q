class Config:
    def __init__(self):
        # calibration settings
        self.optim_size = 1024
        self.calib_size = 128
        self.optim_batch_size = 32
        self.calib_batch_size = 32
        self.w_bit = 4
        self.a_bit = 4
        self.qconv_a_bit = 8
        self.qhead_a_bit = 4
        self.calib_metric = 'mse'       # 'fisher_diag' for Fisher-weighted calibration
        self.matmul_head_channel_wise = True
        self.token_channel_wise = True
        self.eq_n = 128
        self.search_round = 3
        # reconstruction settings
        self.keep_gpu = True
        self.optim_metric = 'fisher_dplr'  # Unified Fisher: DPLR-FIM for AdaRound
        self.temp = 20
        # MLP reconstruction settings
        self.recon_metric = 'fisher_diag'   # Fisher-guided MR: 'fisher_diag' | 'fisher_dplr' | 'mse' | 'mae'
        self.pct = 0.9999                   # clamp percentile for GELU clamping
        # fisher settings (shared across MR, calibration, and BlockRecon)
        self.k = 5
        self.p1 = 1.0
        self.p2 = 1.0
        self.dis_mode = 'q'
        # Adaptive Fisher parameters (SynFIM-Q optimizations)
        self.adaptive_k = True   # Layered dynamic rank (k+3 for early blocks, k-2 for head)
        self.adaptive_p = True   # Adaptive p1/p2 based on block activation std
        self.logit_guard = True  # Full-model logits/confidence guard during block reconstruction
        # qdrop settings
        self.optim_mode = 'qdrop'
        self.drop_prob = 0.5

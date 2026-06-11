class Config:
    def __init__(self):
        # calibration settings
        self.optim_size = 1024
        self.calib_size = 128
        self.optim_batch_size = 32
        self.calib_batch_size = 32
        self.w_bit = 3
        self.a_bit = 3
        self.qconv_a_bit = 8
        self.qhead_a_bit = 3
        self.calib_metric = 'mse'
        self.matmul_head_channel_wise = True
        self.token_channel_wise = True
        self.eq_n = 128
        self.search_round = 3
        # MLP reconstruction settings
        self.recon_metric = 'fisher_diag'   # Fisher-guided MR
        self.pct = 0.9999                   # GELU clamping percentile
        # optimization settings
        self.keep_gpu = True
        self.optim_metric = 'fisher_dplr'
        self.temp = 20
        # fisher settings
        self.k = 5
        self.p1 = 1.0
        self.p2 = 1.0
        self.dis_mode = 'q'
        # Adaptive Fisher parameters
        self.adaptive_k = True   # Layered dynamic rank
        self.adaptive_p = True   # Adaptive p1/p2
        self.adaptive_candidate_select = False
        self.adaptive_candidate_margin = 0.003
        self.logit_guard = True
        # qdrop settings
        self.optim_mode = 'qdrop'
        self.drop_prob = 0.5

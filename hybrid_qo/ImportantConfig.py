import torch
from math import log

class Config:
    def __init__(self, ):
        # self.datafile = 'JOBqueries.workload'
        self.schemaFile = "schema.sql"
        self.user = 'postgres'
        self.password = 'postgres'
        self.dataset = 'tpch'
        self.userName = self.user
        self.usegpu = True
        self.head_num = 10
        self.input_size = 9  # 7+2, 7 = All types (scans/joins), 2= ['total cost', 'plan rows']
        self.hidden_size = 64
        self.batch_size = 256
        self.ip = "pg_balsa"
        self.port = 5432
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cpudevice = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.var_weight = 0.00  # for au, 0:disable,0.01:enable
        self.cost_test_for_debug = False
        self.max_hint_num = 20  # TopN?
        self.max_time_out = 3 * 60 * 1000
        self.threshold = log(3) / log(self.max_time_out)
        self.leading_length = 2
        self.try_hint_num = 3
        self.mem_size = 2000
        self.mcts_v = 1.1
        self.searchFactor = 4
        self.U_factor = 0.0
        self.log_file = 'log_c3_h64_s4_t3.txt'
        self.latency_file = 'latency_record.txt'
        self.modelpath = 'model/'
        self.offset = 20  # Offset for numerical stability?

        # STACK
        # -------------------------------------
        self.database = 'tpch'
        self.n_epochs = 2
        self.max_alias_num = 8
        self.aliasname2id = {
            "customer": 0,
            "lineitem": 1,
            "nation": 2,
            "orders": 3,
            "part": 4,
            "partsupp": 5,
            "region": 6,
            "supplier": 7,
        }
        self.id2aliasname = {value: key for key, value in self.aliasname2id.items()}
        self.max_column = 61

        self.queries_file = 'workload/combined_sql_output__train.json'

        self.mcts_input_size = self.max_alias_num * self.max_alias_num + self.max_column

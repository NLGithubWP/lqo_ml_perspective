from experiments import *

######################### NeurBench (NB) Revision #########################


######################### 1. JOB query on multiple datasets #########################

debug_test_query_glob = ["2a.sql", "2b.sql", "2c.sql", "2d.sql"]
debug_query_dir = '/app/AI4QueryOptimizer/experiment_setup/workloads/balsa/job_query_debug'
debug_val_iters = 1

# define what we used here.
# 1. debug setting
current_used_query_dir = debug_query_dir
current_used_test_query_glob_in_train = debug_test_query_glob
current_used_test_query_glob_in_test = debug_test_query_glob
current_val_iters = 1


@balsa.params_registry.Register
class NB_Balsa_train_imdb_ori_job_debug(Balsa_JOB_EvaluationBase):
    def Params(self):
        p = super().Params()
        p.db = 'imdb_ori'
        # this is the path in docker
        p.query_dir = current_used_query_dir
        p.test_query_glob = current_used_test_query_glob_in_train
        p.validate_every_n_epochs = 200
        p.val_iters = current_val_iters
        p.model_save_path = '/app/AI4QueryOptimizer/experiment_setup/vldb_revision/job/res_balsa/data_shift/client'
        p.model_prefix = 'balsa_imdb_ori_job_mini'
        return p


@balsa.params_registry.Register
class NB_Balsa_train_imdb_01v2_job_debug(Balsa_JOB_EvaluationBase):
    def Params(self):
        p = super().Params()
        p.db = 'imdb_01v2'
        # this is the path in docker
        p.query_dir = current_used_query_dir
        p.test_query_glob = current_used_test_query_glob_in_train
        p.validate_every_n_epochs = 200
        p.val_iters = current_val_iters
        p.model_save_path = '/app/AI4QueryOptimizer/experiment_setup/vldb_revision/job/res_balsa/data_shift/client'
        p.model_prefix = 'balsa_imdb_01v2_job_mini'
        return p


@balsa.params_registry.Register
class NB_Balsa_train_imdb_05v2_job_debug(Balsa_JOB_EvaluationBase):
    def Params(self):
        p = super().Params()
        p.db = 'imdb_05v2'
        # this is the path in docker
        p.query_dir = current_used_query_dir
        p.test_query_glob = current_used_test_query_glob_in_train
        p.validate_every_n_epochs = 200
        p.val_iters = current_val_iters
        p.model_save_path = '/app/AI4QueryOptimizer/experiment_setup/vldb_revision/job/res_balsa/data_shift/client'
        p.model_prefix = 'balsa_imdb_05v2_job_mini'
        return p


@balsa.params_registry.Register
class NB_Balsa_train_imdb_17v2_job_debug(Balsa_JOB_EvaluationBase):
    def Params(self):
        p = super().Params()
        p.db = 'imdb_17v2'
        # this is the path in docker
        p.query_dir = current_used_query_dir
        p.test_query_glob = current_used_test_query_glob_in_train
        p.validate_every_n_epochs = 200
        p.val_iters = current_val_iters
        p.model_save_path = '/app/AI4QueryOptimizer/experiment_setup/vldb_revision/job/res_balsa/data_shift/client'
        p.model_prefix = 'balsa_imdb_17v2_job_mini'
        return p


@balsa.params_registry.Register
class NB_Neo_train_imdb_ori_job_debug(Neo_JOB_EvaluationBase):
    def Params(self):
        p = super().Params()
        p.db = 'imdb_ori'
        # this is the path in docker
        p.query_dir = current_used_query_dir
        p.test_query_glob = current_used_test_query_glob_in_train
        p.validate_every_n_epochs = 200
        p.val_iters = current_val_iters
        p.model_save_path = '/app/AI4QueryOptimizer/experiment_setup/vldb_revision/job/res_neo/data_shift/client'
        p.model_prefix = 'neo_imdb_ori_job_mini'
        return p


@balsa.params_registry.Register
class NB_Neo_train_imdb_01v2_job_debug(Neo_JOB_EvaluationBase):
    def Params(self):
        p = super().Params()
        p.db = 'imdb_01v2'
        # this is the path in docker
        p.query_dir = current_used_query_dir
        p.test_query_glob = current_used_test_query_glob_in_train
        p.validate_every_n_epochs = 200
        p.val_iters = current_val_iters
        p.model_save_path = '/app/AI4QueryOptimizer/experiment_setup/vldb_revision/job/res_neo/data_shift/client'
        p.model_prefix = 'neo_imdb_01v2_job_mini'
        return p


@balsa.params_registry.Register
class NB_Neo_train_imdb_05v2_job_debug(Neo_JOB_EvaluationBase):
    def Params(self):
        p = super().Params()
        p.db = 'imdb_05v2'
        # this is the path in docker
        p.query_dir = current_used_query_dir
        p.test_query_glob = current_used_test_query_glob_in_train
        p.validate_every_n_epochs = 200
        p.val_iters = current_val_iters
        p.model_save_path = '/app/AI4QueryOptimizer/experiment_setup/vldb_revision/job/res_neo/data_shift/client'
        p.model_prefix = 'neo_imdb_05v2_job_mini'
        return p


@balsa.params_registry.Register
class NB_Neo_train_imdb_17v2_job_debug(Neo_JOB_EvaluationBase):
    def Params(self):
        p = super().Params()
        p.db = 'imdb_17v2'
        # this is the path in docker
        p.query_dir = current_used_query_dir
        p.test_query_glob = current_used_test_query_glob_in_train
        p.validate_every_n_epochs = 200
        p.val_iters = current_val_iters
        p.model_save_path = '/app/AI4QueryOptimizer/experiment_setup/vldb_revision/job/res_neo/data_shift/client'
        p.model_prefix = 'neo_imdb_17v2_job_mini'
        return p


######## JOB but test class #########


@balsa.params_registry.Register
class NB_Balsa_test_imdb_ori_job_debug(Balsa_JOB_EvaluationBase):
    def Params(self):
        p = super().Params()
        p.db = 'imdb_ori'
        # this is the path in docker
        p.query_dir = current_used_query_dir
        p.test_query_glob = current_used_test_query_glob_in_test
        return p


@balsa.params_registry.Register
class NB_Balsa_test_imdb_01v2_job_debug(Balsa_JOB_EvaluationBase):
    def Params(self):
        p = super().Params()
        p.db = 'imdb_01v2'
        # this is the path in docker
        p.query_dir = current_used_query_dir
        p.test_query_glob = current_used_test_query_glob_in_test
        return p


@balsa.params_registry.Register
class NB_Balsa_test_imdb_05v2_job_debug(Balsa_JOB_EvaluationBase):
    def Params(self):
        p = super().Params()
        p.db = 'imdb_05v2'
        # this is the path in docker
        p.query_dir = current_used_query_dir
        p.test_query_glob = current_used_test_query_glob_in_test
        return p


@balsa.params_registry.Register
class NB_Balsa_test_imdb_17v2_job_debug(Balsa_JOB_EvaluationBase):
    def Params(self):
        p = super().Params()
        p.db = 'imdb_17v2'
        # this is the path in docker
        p.query_dir = current_used_query_dir
        p.test_query_glob = current_used_test_query_glob_in_test
        return p


@balsa.params_registry.Register
class NB_Neo_test_imdb_ori_job_debug(Neo_JOB_EvaluationBase):
    def Params(self):
        p = super().Params()
        p.db = 'imdb_ori'
        # this is the path in docker
        p.query_dir = current_used_query_dir
        p.test_query_glob = current_used_test_query_glob_in_test
        return p


@balsa.params_registry.Register
class NB_Neo_test_imdb_01v2_job_debug(Neo_JOB_EvaluationBase):
    def Params(self):
        p = super().Params()
        p.db = 'imdb_01v2'
        # this is the path in docker
        p.query_dir = current_used_query_dir
        p.test_query_glob = current_used_test_query_glob_in_test
        return p


@balsa.params_registry.Register
class NB_Neo_test_imdb_05v2_job_debug(Neo_JOB_EvaluationBase):
    def Params(self):
        p = super().Params()
        p.db = 'imdb_05v2'
        # this is the path in docker
        p.query_dir = current_used_query_dir
        p.test_query_glob = current_used_test_query_glob_in_test
        return p


@balsa.params_registry.Register
class NB_Neo_test_imdb_17v2_job_debug(Neo_JOB_EvaluationBase):
    def Params(self):
        p = super().Params()
        p.db = 'imdb_17v2'
        # this is the path in docker
        p.query_dir = current_used_query_dir
        p.test_query_glob = current_used_test_query_glob_in_test
        return p

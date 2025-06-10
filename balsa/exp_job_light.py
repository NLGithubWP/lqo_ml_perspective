from experiments import *
import os
######################### NeurBench (NB) Revision #########################


######################### 1. JOB query on multiple datasets #########################

full_query_query_dir = '/app/AI4QueryOptimizer/experiment_setup/workloads/balsa/job-light-train'
full_query_test_query_glob = [f for f in os.listdir(full_query_query_dir) if os.path.isfile(os.path.join(full_query_query_dir, f))]
empty_test_query_glob = ['2.sql']

current_used_query_dir = full_query_query_dir
current_used_test_query_glob_in_train = empty_test_query_glob
current_used_test_query_glob_in_test = full_query_test_query_glob
# originally, it is 100 for job, 50 for stack datasets
current_val_iters = 10


@balsa.params_registry.Register
class NB_Balsa_train_imdb_ori_job_light(Balsa_JOB_EvaluationBase):
    def Params(self):
        p = super().Params()
        p.db = 'imdb_ori'
        # this is the path in docker
        p.query_dir = current_used_query_dir
        p.test_query_glob = current_used_test_query_glob_in_train
        p.validate_every_n_epochs = 200
        p.val_iters = current_val_iters
        p.model_save_path = '/app/AI4QueryOptimizer/experiment_setup/vldb_revision/job/res_balsa/data_shift/client'
        p.model_prefix = 'balsa_imdb_ori_job_light_full'
        return p


@balsa.params_registry.Register
class NB_Balsa_train_imdb_01v2_job_light(Balsa_JOB_EvaluationBase):
    def Params(self):
        p = super().Params()
        p.db = 'imdb_01v2'
        # this is the path in docker
        p.query_dir = current_used_query_dir
        p.test_query_glob = current_used_test_query_glob_in_train
        p.validate_every_n_epochs = 200
        p.val_iters = current_val_iters
        p.model_save_path = '/app/AI4QueryOptimizer/experiment_setup/vldb_revision/job/res_balsa/data_shift/client'
        p.model_prefix = 'balsa_imdb_01v2_job_light_full'
        return p


@balsa.params_registry.Register
class NB_Balsa_train_imdb_05v2_job_light(Balsa_JOB_EvaluationBase):
    def Params(self):
        p = super().Params()
        p.db = 'imdb_05v2'
        # this is the path in docker
        p.query_dir = current_used_query_dir
        p.test_query_glob = current_used_test_query_glob_in_train
        p.validate_every_n_epochs = 200
        p.val_iters = current_val_iters
        p.model_save_path = '/app/AI4QueryOptimizer/experiment_setup/vldb_revision/job/res_balsa/data_shift/client'
        p.model_prefix = 'balsa_imdb_05v2_job_light_full'
        return p


@balsa.params_registry.Register
class NB_Balsa_train_imdb_17v2_job_light(Balsa_JOB_EvaluationBase):
    def Params(self):
        p = super().Params()
        p.db = 'imdb_17v2'
        # this is the path in docker
        p.query_dir = current_used_query_dir
        p.test_query_glob = current_used_test_query_glob_in_train
        p.validate_every_n_epochs = 200
        p.val_iters = current_val_iters
        p.model_save_path = '/app/AI4QueryOptimizer/experiment_setup/vldb_revision/job/res_balsa/data_shift/client'
        p.model_prefix = 'balsa_imdb_17v2_job_light_full'
        return p


@balsa.params_registry.Register
class NB_Neo_train_imdb_ori_job_light(Neo_JOB_EvaluationBase):
    def Params(self):
        p = super().Params()
        p.db = 'imdb_ori'
        # this is the path in docker
        p.query_dir = current_used_query_dir
        p.test_query_glob = current_used_test_query_glob_in_train
        p.validate_every_n_epochs = 200
        p.val_iters = current_val_iters
        p.model_save_path = '/app/AI4QueryOptimizer/experiment_setup/vldb_revision/job/res_neo/data_shift/client'
        p.model_prefix = 'neo_imdb_ori_job_light_full'
        return p


@balsa.params_registry.Register
class NB_Neo_train_imdb_01v2_job_light(Neo_JOB_EvaluationBase):
    def Params(self):
        p = super().Params()
        p.db = 'imdb_01v2'
        # this is the path in docker
        p.query_dir = current_used_query_dir
        p.test_query_glob = current_used_test_query_glob_in_train
        p.validate_every_n_epochs = 200
        p.val_iters = current_val_iters
        p.model_save_path = '/app/AI4QueryOptimizer/experiment_setup/vldb_revision/job/res_neo/data_shift/client'
        p.model_prefix = 'neo_imdb_01v2_job_light_full'
        return p


@balsa.params_registry.Register
class NB_Neo_train_imdb_05v2_job_light(Neo_JOB_EvaluationBase):
    def Params(self):
        p = super().Params()
        p.db = 'imdb_05v2'
        # this is the path in docker
        p.query_dir = current_used_query_dir
        p.test_query_glob = current_used_test_query_glob_in_train
        p.validate_every_n_epochs = 200
        p.val_iters = current_val_iters
        p.model_save_path = '/app/AI4QueryOptimizer/experiment_setup/vldb_revision/job/res_neo/data_shift/client'
        p.model_prefix = 'neo_imdb_05v2_job_light_full'
        return p


@balsa.params_registry.Register
class NB_Neo_train_imdb_17v2_job_light(Neo_JOB_EvaluationBase):
    def Params(self):
        p = super().Params()
        p.db = 'imdb_17v2'
        # this is the path in docker
        p.query_dir = current_used_query_dir
        p.test_query_glob = current_used_test_query_glob_in_train
        p.validate_every_n_epochs = 200
        p.val_iters = current_val_iters
        p.model_save_path = '/app/AI4QueryOptimizer/experiment_setup/vldb_revision/job/res_neo/data_shift/client'
        p.model_prefix = 'neo_imdb_17v2_job_light_full'
        return p


######## JOB but test class #########


@balsa.params_registry.Register
class NB_Balsa_test_imdb_ori_job_light(Balsa_JOB_EvaluationBase):
    def Params(self):
        p = super().Params()
        p.db = 'imdb_ori'
        # this is the path in docker
        p.query_dir = current_used_query_dir
        p.test_query_glob = current_used_test_query_glob_in_test
        return p


@balsa.params_registry.Register
class NB_Balsa_test_imdb_01v2_job_light(Balsa_JOB_EvaluationBase):
    def Params(self):
        p = super().Params()
        p.db = 'imdb_01v2'
        # this is the path in docker
        p.query_dir = current_used_query_dir
        p.test_query_glob = current_used_test_query_glob_in_test
        return p


@balsa.params_registry.Register
class NB_Balsa_test_imdb_05v2_job_light(Balsa_JOB_EvaluationBase):
    def Params(self):
        p = super().Params()
        p.db = 'imdb_05v2'
        # this is the path in docker
        p.query_dir = current_used_query_dir
        p.test_query_glob = current_used_test_query_glob_in_test
        return p


@balsa.params_registry.Register
class NB_Balsa_test_imdb_17v2_job_light(Balsa_JOB_EvaluationBase):
    def Params(self):
        p = super().Params()
        p.db = 'imdb_17v2'
        # this is the path in docker
        p.query_dir = current_used_query_dir
        p.test_query_glob = current_used_test_query_glob_in_test
        return p


@balsa.params_registry.Register
class NB_Neo_test_imdb_ori_job_light(Neo_JOB_EvaluationBase):
    def Params(self):
        p = super().Params()
        p.db = 'imdb_ori'
        # this is the path in docker
        p.query_dir = current_used_query_dir
        p.test_query_glob = current_used_test_query_glob_in_test
        return p


@balsa.params_registry.Register
class NB_Neo_test_imdb_01v2_job_light(Neo_JOB_EvaluationBase):
    def Params(self):
        p = super().Params()
        p.db = 'imdb_01v2'
        # this is the path in docker
        p.query_dir = current_used_query_dir
        p.test_query_glob = current_used_test_query_glob_in_test
        return p


@balsa.params_registry.Register
class NB_Neo_test_imdb_05v2_job_light(Neo_JOB_EvaluationBase):
    def Params(self):
        p = super().Params()
        p.db = 'imdb_05v2'
        # this is the path in docker
        p.query_dir = current_used_query_dir
        p.test_query_glob = current_used_test_query_glob_in_test
        return p


@balsa.params_registry.Register
class NB_Neo_test_imdb_17v2_job_light(Neo_JOB_EvaluationBase):
    def Params(self):
        p = super().Params()
        p.db = 'imdb_17v2'
        # this is the path in docker
        p.query_dir = current_used_query_dir
        p.test_query_glob = current_used_test_query_glob_in_test
        return p

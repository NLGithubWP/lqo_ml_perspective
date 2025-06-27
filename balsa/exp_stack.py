from experiments import *

stack_full_query = [
    ["test_only",
     "q1__q1-009.sql",
     "q1__q1-031.sql",
     "q1__q1-035.sql",
     "q1__q1-067.sql",
     "q1__q1-075.sql",
     "q1__q1-098.sql",
     "q1__q1-099.sql",
     "q1__q1-100.sql",
     "q2__q2-001.sql",
     "q2__q2-012.sql",
     "q2__q2-032.sql",
     "q2__q2-035.sql",
     "q2__q2-050.sql",
     "q2__q2-081.sql",
     "q2__q2-094.sql",
     "q2__q2-098.sql",
     "q3__q3-018.sql",
     "q3__q3-040.sql",
     "q3__q3-043.sql",
     "q3__q3-046.sql",
     "q3__q3-066.sql",
     "q3__q3-068.sql",
     "q3__q3-086.sql",
     "q3__q3-099.sql",
     "q4__q4-002.sql",
     "q4__q4-026.sql",
     "q4__q4-041.sql",
     "q4__q4-042.sql",
     "q4__q4-064.sql",
     "q4__q4-074.sql",
     "q4__q4-086.sql",
     "q4__q4-089.sql",
     "q5__q5-015.sql",
     "q5__q5-032.sql",
     "q5__q5-041.sql",
     "q5__q5-052.sql",
     "q5__q5-059.sql",
     "q5__q5-077.sql",
     "q5__q5-079.sql",
     "q5__q5-082.sql",
     "q6__q6-002.sql",
     "q6__q6-009.sql",
     "q6__q6-060.sql",
     "q6__q6-064.sql",
     "q6__q6-065.sql",
     "q6__q6-067.sql",
     "q6__q6-069.sql",
     "q6__q6-085.sql",
     "q7__q7-034.sql",
     "q7__q7-036.sql",
     "q7__q7-047.sql",
     "q7__q7-077.sql",
     "q7__q7-082.sql",
     "q7__q7-085.sql",
     "q7__q7-095.sql",
     "q7__q7-099.sql",
     "q8__q8-006.sql",
     "q8__q8-025.sql",
     "q8__q8-046.sql",
     "q8__q8-062.sql",
     "q8__q8-065.sql",
     "q8__q8-074.sql",
     "q8__q8-076.sql",
     "q8__q8-096.sql",
     "q11__0ea8bacde0e13a4314466435cf49c8e685b39fb1.sql",
     "q11__6c5cba419c5b7b02d431aeb5e766d775d812967a.sql",
     "q11__33e1caf220e5bea2e592c82eede1c0427e2c2570.sql",
     "q11__87c4bd0930b02a3361ac2e86c453db1fec60dc6b.sql",
     "q11__9389f58853715321e2a60ad743f99fc365f040cb.sql",
     "q11__aa96c8d7abbf8a5b6d29473c1b9447a84f8b4f52.sql",
     "q11__c1ae2a992cde4ea2c4922d852df22043254b4f84.sql",
     "q11__e4ca35591923cf4efc89e64b17fb4d330c0b34df.sql",
     "q12__5a5ff9bd9de9e748708116727803117e453e30da.sql",
     "q12__06c8d6886a03d4d92837f38ff395b888de007d33.sql",
     "q12__55de941e8497cfeeb93d3f8f2d7a18489e0e6c32.sql",
     "q12__76a47868a09eec9f95bacb2cf21492d353698eb7.sql",
     "q12__547c6bf1994c9b2ba82a7ae32f4b051beabf46fd.sql",
     "q12__812a3effb91cb789490fc2e12af772b1a35f8552.sql",
     "q12__0700720596313f7fa30c0dd3d4a3001c896ba760.sql",
     "q12__bde6c0cf5e67ddae4ec0dbb787291da703e406d5.sql",
     "q13__1ddcc8650e17b292bc7344902baffc90c5ae5761.sql",
     "q13__13ad1b8c6bea4fda1892b9fa82cc1ceb9ceb85fc.sql",
     "q13__935e2051bf80eeafe91aeb6eb719b6b64b9592c2.sql",
     "q13__a3d03772d880754fc4e150d82908757477ae2186.sql",
     "q13__a091adce62743b65c04532e98e8ff3d7e546ea77.sql",
     "q13__add0df9dccb2790c14508e19c9e0deb79fad6ea2.sql",
     "q13__d383cd5b4aee7d3f73508e2a1fe5f6d0f7dd42a2.sql",
     "q13__d4707be2adfdbc842f42acb1fc16e3a43faf7474.sql",
     "q14__5dbc1d1f1a0467ad0086e6cb337144387a37533a.sql",
     "q14__5e4835cd72aaa2d7be15b2a5ffa2e66156b3656f.sql",
     "q14__63c0776f1727638316b966fe748df7cc585a335b.sql",
     "q14__74fd1af68d23f0690e3d0fc80bd9b42fa90a7e94.sql",
     "q14__97e68ad5c2ced4c182366b3118a1f5f69b423fa6.sql",
     "q14__719e692d411868ae7a93909757872d264f6bbf73.sql",
     "q14__4063b6cbbd1c0f2a902a647aafe24174a75f53cd.sql",
     "q14__b49361f85785200ed6ec1f2eec357b7598c9e564.sql",
     "q15__3e37e62655ceaebc14e79edad518e5710752f51d.sql",
     "q15__21e4988a3f47be288de5891d69acf91928ed94eb.sql",
     "q15__543ab3f730e494a69e3d15e59675f491544cb15d.sql",
     "q15__78995a5fc0536aa53b99be32ce84dcbf40e826f3.sql",
     "q15__b2ee2c788d30655058aeb992811e9a54f17f2998.sql",
     "q15__b8ddf65b0c0c7867a9b560e571d457fec410715c.sql",
     "q15__c9619ad44302bada330d337c174f9dab77538622.sql",
     "q15__d5546c01928a687eb1f54e9f8eb4e1aff68fc381.sql",
     "q16__1e863562a79ca1f7754c759ebab6a2addda0bde8.sql",
     "q16__374e3e4c9eefc294fa4c46220953336298df3622.sql",
     "q16__b1a96cd48ba297dd93bce73c27b491069ad7449f.sql",
     "q16__d5290889129fb8e625f2b36fa106e30d6c4b243b.sql",
     "q16__ea9efde510227beb8d624b8c4a6941b9d5e6e637.sql",
     "q16__ed2ffeaefcf5ad8bbadc713ccc766541e12080aa.sql",
     "q16__f67cec3d635586efb847c832072be83b42cc45b7.sql",
     "q16__fbe34e8fdf672a34fd82cbbd6d9a81fd02ce17d1.sql"]
]

stack_full_query_query_dir = 'queries/stack'
stack_empty_test_query_glob = ['test_only.sql']
stack_current_val_iters = 50


@balsa.params_registry.Register
class NB_Balsa_train_stack08(Balsa_STACK_EvaluationBase):
    def Params(self):
        p = super().Params()
        p.db = 'stack2008'
        # this is the path in docker
        p.query_dir = stack_full_query_query_dir
        p.test_query_glob = stack_empty_test_query_glob
        p.validate_every_n_epochs = 200
        p.val_iters = stack_current_val_iters
        p.model_save_path = '/hdd1/xingnaili/AI4QueryOptimizer/baseline/lqo_ml_perspective/balsa/trained_models'
        p.model_prefix = 'balsa_stack2008_so_112'
        return p


@balsa.params_registry.Register
class NB_Balsa_test_stack08(Balsa_STACK_EvaluationBase):
    def Params(self):
        p = super().Params()
        p.db = 'stack2008'
        p.query_dir = stack_full_query_query_dir
        p.test_query_glob = stack_full_query_query_dir
        return p


@balsa.params_registry.Register
class NB_Balsa_test_stack10(Balsa_STACK_EvaluationBase):
    def Params(self):
        p = super().Params()
        p.db = 'stack2010'
        p.query_dir = stack_full_query_query_dir
        p.test_query_glob = stack_full_query_query_dir
        return p

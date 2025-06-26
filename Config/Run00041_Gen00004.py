import math

# Example config for generation with CFG (M4)

class data_params:
    data_dir="/gdata/projects/ml_scope/Turbulence/QG_V0001/Results/"
    file_prefix = "fields_pooled64_Run"
    batch_size=64
    run_start = 2727
    run_end = 3220
    min_t = 101 
    cond_data_file_prefix = "fields_pooled64_ds16_partial_Run"
    field_std = 1.3
    
class sde_params:
    time_steps= 64
    option = 0 # 0 is cosine, 1 is linear noise scheduling
    
class train_params:
    num_epochs = 1025
    learning_rate  = 0.0002 

class network_params:
    channels = [32,64,128,256]
    in_channels=2
    out_channels=1
    dropout_rate=0
    attention = False
    condition = True
    
class gen_params:
    run_number = 3221
    snapshot = 81
    seed = 10
    num_samples = 16
    num_corrections =2
    corr_scale = 0.3
    model_uncond=32
    checkpoint = 250
    # SDEdit
    class sdedit:
        flag = False
        time = 64
    # DPS
    class DPS:
        flag          = False
        scale         = 4
        C             = 2e-5
        sigma_measure = 1e-2
        gappy         = False
    # Vanilla cond
    cond_flag= True
    model_cond=41
    # CFG
    class CFG:
        flag   = True
        weight = 1.25
        
class params:
    gen = gen_params
    data= data_params
    sde = sde_params
    train = train_params
    network= network_params
    run_number = 41
    gen_number =4
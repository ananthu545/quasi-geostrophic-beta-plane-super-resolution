import math

# Example config for conditional training

class data_params:
    data_dir="/gdata/projects/ml_scope/Turbulence/QG_V0001/Results/"
    file_prefix = "fields_pooled64_Run"
    batch_size=64
    # QG simulation runs used for training
    run_start = 2727
    run_end = 3220
    # Snapshot starting index (after self-similarity)
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
    
class params:
    data= data_params
    sde = sde_params
    train = train_params
    network= network_params
    run_number = 41
    

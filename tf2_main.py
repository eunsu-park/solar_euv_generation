name_model = 'UNET'
layer_max_d = 3
isize = 1024

gpu_id = '0'

instr_input = 'hmi'
instr_output = 'aia'

type_input = 'M_720s'
type_output = '304'

ch_input = 1
ch_output = 1
ch_axis = -1

bsize = 1
shake = True

iter_display = 2000
iter_save = 10000
iter_max = 500000

root_data = '/userhome/park_e/datasets'
root_save = '/userhome/park_e/solar_euv_generation'

root_ckpt = '%s/%s/%s/ckpt'%(root_save, version, mode)
root_snap = '%s/%s/%s/snap'%(root_save, version, mode)
root_test = '%s/%s/%s/test'%(root_save, version, mode)

class data_generator():
    def __init__(self, {}) :





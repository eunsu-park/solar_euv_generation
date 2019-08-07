from utils import make_tensor, shake_tensor, make_output

class option_base():
    def __init__(self):
        
        print('\n------------ Options for base ------------\n')
        
        name_models = ['UNET', 'SUNET', 'SSUNET', 'CSSUNET']
        print('\nAvailable Model: UNET, SUNET, SSUNET, CSSUNET\n')
        answer = input('Select Model?     ')
        if answer.upper() not in name_models :
            raise NameError('%s: Invalid model name'%(answer))
        self.name_model = answer.upper()
        del answer

        print('\n# of layer in PatchGAN Discriminator (Receptive Field Size): 0(1), 1(16), 2(34), 3(70), 4(142), 5(286)\n')
        answer = input('# of layers?     ')
        if int(answer) not in range(6):
            raise ValueError('%s: Invalid # of layers in Discriminator'%(answer))
        self.layer_max_d = int(answer)
        del answer

        wavelnths = ['94', '131', '171', '193', '211', '304', '335', '1600', '1700']
        print('\nPossible AIA wavelengths: 94, 131, 171, 193, 211, 304, 335, 1600, 1700\n')
        answer = str(int(input('AIA wavelength?     ')))
        if answer not in wavelnths :
            raise ValueError('%s: Invalid AIA wavelength'%(answer))
        self.wavelnth = answer
        self.name_input, self.name_output = 'M_720s', '%s'%(self.wavelnth)
        self.ch_input, self.ch_output = 1, 1
        self.instr_input, self.instr_output = 'hmi', 'aia'
        del answer
        
        self.isize = 1024
        self.ch_axis = -1

        self.mode = '%s_%s.%s_%s'%(self.instr_input, self.name_input, self.instr_output, self.name_output)
        self.version = 'CGAN_%s_%d_%dD'%(self.name_model, self.isize, self.layer_max_d)
        
        self.root_data = '/userhome/park_e/datasets'
        self.root_save = '/userhome/park_e/solar_euv_generation'

        self.root_ckpt = '%s/%s/%s/ckpt'%(self.root_save, self.version, self.mode)
        self.root_snap = '%s/%s/%s/snap'%(self.root_save, self.version, self.mode)
        self.root_test = '%s/%s/%s/test'%(self.root_save, self.version, self.mode)
        
        self.make_tensor_input = make_tensor(self.isize, is_aia=False)
        self.make_tensor_output = make_tensor(self.isize, is_aia=True)
        self.shake_tensor = shake_tensor(self.isize)
        self.make_output = make_output(self.isize, self.wavelnth)
        
class option_train(option_base):        
    def __init__(self):
        super(option_train, self).__init__()
        
        print('\n------------ Options for train ------------\n')
        
        self.gpu_id = input('GPU ID?     ')
        self.iter_display = int(input('Display frequency(iter)?     '))
        self.iter_save = int(input('Save frequency(iter)?     '))
        self.iter_max = int(input('Max iteration?     '))
        
        self.bsize = 1        
        self.shake=True            
                
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

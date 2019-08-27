# Preprocessing

### 1.image_processing
### 2.list_processing
### 3.augmentation_3D
### 4.gui

<code>
  <p>
    
    !rm -r Keeplearning
    !rm -r Pytorch_library
    !rm -r Keras_library

    !git clone https://github.com/kevinkwshin/Keeplearning.git
    !git clone https://github.com/kevinkwshin/Pytorch_library.git
    !git clone https://github.com/kevinkwshin/Keras_library.git

    # from Pytorch_library.losses import *
    # from Pytorch_library.models_3D import *

    from Keras_library.keras_preprocessing_3D import *

    from Keeplearning.image_processing import *
    from Keeplearning.list_processing import *
    from Keeplearning.load_data import *
    from Keeplearning.augmentation_3D import *
    from Keeplearning.visualize import *
    from Keeplearning.metrics import *
    
  </p>
</code>

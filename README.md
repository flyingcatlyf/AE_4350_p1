# AE_4350_p1
The 1st part of code for course AE-4350: ddpg-sca algorithm
# file content
A) ddpg-sca training
    0) Training.py  -Main file for training the flight control system (run to start the training process)
    1) RK.py  -iteratively solves lateral dynamical equations over time with Runge Kutta method
    2) Ref.py -Producecs reference signals
    3) ddpg_torch.py  -Synthesizes all functions of the DDPG-SCA agent
# guide to use the code
A) RUN TRAINING
    0) Install an interpreter environment that includes packages in interpreter_packages.txt
    1) Run Training.py in files 'DDPG-SCA_Training', 'DDPG-SDA_Training', 'DDPG_Training', to start training of the flight controller.  
    2) Save data automatically in file 'tmp/ddpg'

B) RUN OPERATION
    0) Install an interpreter environment that includes packages in interpreter_packages.txt
    1) Run Operation.py in file 'Operation&Plot', to start operation phase. The file 'Actor_ddpg.zip' includes actor weights file and should be uploaded to run operation.
    2) Save data automatically

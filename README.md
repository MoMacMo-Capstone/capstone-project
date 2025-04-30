# SeismicGAN
Nathan Koppel, Vishnu Naidu, Aaron Weng
## What is SeismicGAN?
- Uses a generative adversarial network (GAN) to inpaint holes in seismic volumes
- Takes in 3d volumes of seismic data in .bin or .segy format
- Generates predictions of missing data
- Reconstructs the 3d volume by filling in each 2d slice individually
- Outputs in .segy file format

## Purpose
- Carbon Sequestration
- Carbon Capture and Storage (CCS)
- Before carbon is released from a power station into the atmosphere, it can be “captured” and sent underground to prevent atmospheric pollution.
- Good underground scans allow companies to know where it is best to store carbon to decrease leakage and other risks 


# Instructions
run for neccessary installs (first time): 
>pip install numpy torch matplotlib segyio torchvision torchmetrics tensorboard

Initial step:  
HOW TO INFILL  
-- To draw a mask and then infill on a select volume from a .bin or .sgy file  
-- Have all files and "trained_64x64_305ee13_div1.ckpt" or another checkpoint in the directory  
-- Default works on "trained_64x64_305ee13_div1.ckpt" but can change to another infilling model  
-- Default works on "SegActi-45x201x201x614.bin" but can change to another collection of volumes in .bin or .segy format(if using bin change variable bin_filename in pipe.py)  

run:  
>python "pipe.py"

-- Select bin or sgy, and if sgy provided file name  
-- select volume index if applicable  
-- Draw a mask using a mouse, then close pop-up window when done  
-- Wait for a little bit (likely around a minute)  
-- See results!  
-- .sgy file will be generated and saved to reconstruct folder in the current directory  

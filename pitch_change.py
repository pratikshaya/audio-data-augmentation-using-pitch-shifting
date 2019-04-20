from __future__ import print_function
import numpy as np
import librosa
import os

file = "E:\\recovery\\work\\data_augumentation_audio\\Samples\\rough\\"

def augment_data(y, sr, n_augment = 0, allow_pitch = True,  tab=""):
    
    mods = [y]                  # always returns the original as element zero

    for i in range(n_augment):
        print(tab+"augment_data: ",i+1,"of",n_augment)
        y_mod = y
        count_changes = 0
        
        # change pitch (w/o speed)
        if (allow_pitch):   
            bins_per_octave = 24        # pitch increments are quarter-steps
            pitch_pm = 4                                # +/- this many quarter steps
            pitch_change =  pitch_pm * 2*(np.random.uniform()-0.5)   
            print(tab+"    pitch_change = ",pitch_change)
            y_mod = librosa.effects.pitch_shift(y, sr, n_steps=pitch_change, bins_per_octave=bins_per_octave)
            count_changes += 1
            
        if (0 == count_changes):
            print("No changes made to signal, trying again")
            mods.append(  augment_data(y, sr, n_augment = 1, tab="      ")[1] )
        else:
            mods.append(y_mod)

    return mods

def main(file):
    np.random.seed(1)
    N = 1
    

    # read in every file on the list, augment it lots of times, output all those
    for infile in os.listdir(file):
        print("Operating on file",infile,".  Requesting ",N," mods...")
        y, sr = librosa.load(infile, sr=None)
        mods = augment_data(y, sr, n_augment=N)
        for i in range(len(mods)-1):
            filename_no_ext = os.path.splitext(infile)[0]
            ext = os.path.splitext(infile)[1]
            outfile = filename_no_ext+"_aug"+str(i+1)+ext
            print("      mod = ",i+1,": saving file",outfile,"...")
            librosa.output.write_wav(outfile,mods[i+1],sr)
    else:
        print(" *** File",infile,"does not exist.  Skipping.")

if __name__ == "__main__":
    main(file)
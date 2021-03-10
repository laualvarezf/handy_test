import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np


## LOADING THE XTEST FOR ONE SUBJECT
def loading_data(subject_number):
    print('loading file')
    xtest_loaded = pickle.load(open(f"HandMotions/data/xtest{subject_number}_50.pkl","rb"))
    print('loaded correctly')
    return xtest_loaded

## VISUALIZING XTEST FOR ONE SUBJECT AT CHOSEN START POINT FOR ONE SECOND

def visualize(second_segment_number, subject_number):
    xtest = loading_data(subject_number)
    channels = (np.transpose(xtest))
    fig= plt.figure(figsize=(40,10))
    plt.tick_params(axis='both', which='major', labelsize=40)
    plt. xlabel('time (ms)', fontsize=50)
    plt. ylabel('Frequency (Hz)', fontsize=50)
    motor_cortex_channels = [4, 8, 9, 12, 13, 14, 18, 19]
    for i in motor_cortex_channels:
        plt.plot(channels[i][(second_segment_number*1000):((second_segment_number*1000)+1000)])
        plt.xlim(0, 1000)
    return st.pyplot(fig)

## FUNCTION FOR SUBJECT DROPDOWN

def get_select_subject():
    print('get_select_box_data called')
    return pd.DataFrame({
          'first column': list(range(1, 13)),
        })

## START OF THE APP
st.markdown('''

    # Welcome to Biohackers Wave Wizard

    ''')


df = get_select_subject()

subject_number = st.selectbox('Select a subject', df['first column'])




second_segment_number = st.slider('second', 0, 50, 0)
print(second_segment_number)
visualize(second_segment_number, subject_number)




'''You chose segment'''
st.write(second_segment_number)

def search(subject_number, second_segment_number):
   pred = pickle.load(open(f"HandMotions/predictions/pred{subject_number}.pkl","rb"))
   return pred[int(second_segment_number)]

predictions= search(subject_number, second_segment_number)
for prediction in predictions:
    st.write(prediction)




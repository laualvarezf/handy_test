import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


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
st.markdown("<h1 style='text-align: center; color: black;'>Welcome to Biohackers Wave Wizard</h1><br>", unsafe_allow_html=True)


st.markdown("<h2 style='text-align: center; color: black;'>How does it work?</h2>", unsafe_allow_html=True)

st.markdown("<p style='text-align: center; color: black;'> Use our app to predict the hand movement of different subjects at a given second. First start by selecting a subject from the dropdown. Afterwards, use the slider to select a second. A visual representation of the main brain waves that control movement will be displayed along with our hand movement prediction. Enjoy! </p><br><br>", unsafe_allow_html=True)

df = get_select_subject()

st.markdown('''
### Please select a subject
''')

subject_number = st.selectbox('1-12', df['first column'])

st.markdown('''
### Please select a second you want to predict on
''')

second_segment_number = st.slider('0-50', 0, 50, 1)
print(second_segment_number)
visualize(second_segment_number, subject_number)


def search(subject_number, second_segment_number):
   pred = pickle.load(open(f"HandMotions/predictions/pred{subject_number}.pkl","rb"))
   return pred[int(second_segment_number)]

st.markdown('''
### Predicted movement
''')

predictions= search(subject_number, second_segment_number)

if len(predictions) == 1:
    if second_segment_number == 1:
        st.markdown(f'At {second_segment_number} second the predicted movement is: **{predictions[0]}**')
    else:
        st.markdown(f'At {second_segment_number} seconds the predicted movement is: **{predictions[0]}**')
else:
    if second_segment_number == 1:
        st.markdown(f'At {second_segment_number} second the predicted movements are: **{predictions[0]}** and **{predictions[-1]}**')
    else:
        st.markdown(f'At {second_segment_number} seconds the predicted movements are: **{predictions[0]}** and **{predictions[-1]}**')


images = []
for name in ['josh', 'kenza', 'sofia', 'yo']:
    image = Image.open(f'images/{name}.jpeg')
    image = image.resize((174, 174))
    images.append(image)

st.markdown("")
st.markdown("<h2 style='text-align: center; color: black;'><br>Meet the team</h2><br>", unsafe_allow_html=True)

st.image(images, caption=['Joshua Bugg', 'Kenza Mandri', 'Sofia Martins', 'Laura Alvarez'], use_column_width=False)

st.markdown("<p style='text-align: center; color: black;'> We are a group of students at Le Wagon London interested in the way we can solve real world problems using data. We hope you enjoy our project as much as we enjoyed making it. Keep posted for updates and have fun!</p>", unsafe_allow_html=True)


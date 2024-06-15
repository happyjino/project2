import tensorflow as tf
from keras.models import Sequential
from keras import layers
from tensorflow.keras.utils import to_categorical 
from keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import datasets
import random
import os
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import keras
from keras.layers import Dropout
###############################################################################################################

save_dir = './model'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

###############################################################################################################

def load_and_preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.convert('L')  # 흑백 변환
    img = img.resize((28, 28))  # 크기 조정
    img_array = np.array(img, dtype=np.float32)
    img_array = img_array / 255.0  # 정규화
    img_array = img_array.reshape((28, 28, 1))  # reshape to (28, 28, 1)
    
    return img_array

def random_invert_img(x):
    noise_factor = 0.3  # 노이즈 강도 조절
    noise = np.random.randn(*x.shape) * noise_factor
    x = x + noise
    x = np.clip(x, 0., 1.)  # 이미지 값을 [0, 1] 범위로 제한
    if random.random() < 0.5:  # 50% 확률로 흑백 전환
        x = 1 - x
    return x


###############################################################################################################

def Data_show():
    ### MNIST data 확인
    ###grey_scale_image
    for num in range(1,10,1):
        plt.subplot(3,3,num)
        plt.imshow(x_train[num-1],cmap='gray')
        plt.title(f'label = [{y_train[num-1]}]')
        plt.axis('off')
    plt.show()

###############################################################################################################

def CNN_classifier(x_train, x_test, y_train, y_test):
    
    datagen = ImageDataGenerator(
        rotation_range=90,     # 무작위로 회전
        width_shift_range=0.1, # 무작위로 수평 이동
        height_shift_range=0.1,# 무작위로 수직 이동
        shear_range=0.5,       # 무작위로 전단 변환
        zoom_range=0.5,        # 무작위로 확대/축소
        fill_mode='nearest',    # 빈 픽셀을 채우는 방법
        preprocessing_function=random_invert_img
    )
    ### 이미지 처리를 위한 데이터 전처리
    x_train = x_train.reshape((-1,28,28,1))/255
    x_test = x_test.reshape((-1,28,28,1))/255

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    # augmented_images = datagen.flow(x_train, y_train, batch_size=1)
        
    # plt.figure(figsize=(10, 10))
    # for i in range(5):
    #     augmented_image = next(augmented_images)[0]
    #     plt.subplot(1, 5, i + 1)
    #     plt.imshow(augmented_image.squeeze(), cmap='gray')
    #     plt.axis('off')
    # plt.show()
    

    ### 학습 구조 만들기
    ##모델 구성 표현_1
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(32, kernel_size=(3,3), input_shape=(28,28,1), padding='same',activation='relu'))
    model.add(layers.MaxPool2D(pool_size=(2,2),padding='same'))
    model.add(Dropout(0.25))
    model.add(layers.Conv2D(64, kernel_size=(3,3), input_shape=(28,28,1), padding='same',activation='relu'))
    model.add(layers.MaxPool2D(pool_size=(2,2),padding='same'))
    model.add(Dropout(0.25))  
    model.add(layers.Conv2D(128, kernel_size=(3,3), input_shape=(28,28,1), padding='same',activation='relu'))
    model.add(layers.MaxPool2D(pool_size=(2,2),padding='same'))
    model.add(Dropout(0.25))  

    model.add(layers.Flatten())
    model.add(layers.Dense(512,activation='relu'))
    model.add(Dropout(0.25))  
    model.add(layers.Dense(10,activation='softmax'))

    ##모델 구성 표현_2
    # model = tf.keras.Sequential([
    #     layers.Conv2D(32, kernel_size=(3,3), input_shape=(28,28,1), padding='same',activation='relu'),
    #     layers.MaxPool2D(pool_size=(2,2),padding='same'),
    #     layers.Conv2D(64, kernel_size=(3,3), padding='same',activation='relu'),
    #     layers.MaxPool2D(pool_size=(2,2),padding='same'),
    #     layers.Flatten(),
    #     layers.Dense(512,activation='relu'),
    #     layers.Dense(10, activation='softmax')
    # ]

    model.compile(optimizer = tf.optimizers.Adam(0.0005), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    
    datagen.fit(x_train)
    model.fit(datagen.flow(x_train, y_train, batch_size=150), epochs=40, validation_data = (x_test, y_test),verbose=1)

    print('Train Accuracy = %f' %model.evaluate(x_train,y_train)[1])
    print('Validation Accuracy = %f' %model.evaluate(x_test,y_test)[1])

    model.save(f'{save_dir}/CNN_classifier_model.keras')

###############################################################################################################

def Evaluation(test_image_path):
    model = tf.keras.models.load_model(f'{save_dir}/CNN_classifier_model.keras')
    image = load_img(test_image_path, target_size = (28,28), color_mode='grayscale')
    img = img_to_array(image)
    img = img/255
    img = np.expand_dims(img, axis = 0)
        
    predictions = model.predict(img)
    print(test_image_path)
    for i in range(10):
        formatted_number = f"{predictions[0][i]:.3f}"
        print(str(i) + ": " + formatted_number)
    predicted_class = np.argmax(predictions)

    plt.imshow(image,cmap='gray')
    plt.title(f'The predicted class of this image is "{predicted_class}"')
    plt.show()

###############################################################################################################

def image_evaluation(model, test_image_path):
    test_image = load_img(test_image_path)

    img = test_image.reshape(-1,28,28,1)/255
    pred = model.predict(img)
    print('Image_evaluation_result = ', pred)

    plt.imshow(test_image)
    plt.axis('off')
    # plt.title(decoding_result(pred))
    plt.show()
 

###############################################################################################################

def Evaluation2(image):
    model = tf.keras.models.load_model(f'{save_dir}/CNN_classifier_model.keras')
    test_image = image.reshape(-1,28,28,1)/255   

    predictions = model.predict(test_image)
    for i in range(10):
        formatted_number = f"{predictions[0][i]:.3f}"
        print(str(i) + ": " + formatted_number)
    predicted_class = np.argmax(predictions)

    plt.imshow(image,cmap='gray')
    plt.title(f'The predicted class of this image is "{predicted_class}"')
    plt.show()

###############################################################################################################

train_image_path = './my_writing_dataset/Training'
jpeg_images = []
jpeg_labels = []

for label in os.listdir(train_image_path):
    label_path = os.path.join(train_image_path, label)
    if os.path.isdir(label_path):
        for file_name in os.listdir(label_path):
            if file_name.endswith('.JPG') or file_name.endswith('.jpg'):
                image_path = os.path.join(label_path, file_name)
                img_array = load_and_preprocess_image(image_path)
                jpeg_images.append(img_array)
                jpeg_labels.append(int(label))

jpeg_images = np.array(jpeg_images)
jpeg_labels = np.array(jpeg_labels)
jpeg_images = jpeg_images.reshape(-1, 28, 28)

jpeg_images = np.repeat(jpeg_images, 300, axis=0)
jpeg_labels = np.repeat(jpeg_labels, 300)


### 코드 실행부
(x_train,y_train), (x_test, y_test) = datasets.mnist.load_data()
x_train = x_train.reshape(60000,28,28)
x_test = x_test.reshape(10000,28,28)

x_train_combined = np.concatenate((x_train, jpeg_images), axis=0)
y_train_combined = np.concatenate((y_train, jpeg_labels), axis=0)

test_image_path = './my_writing_dataset/test'

# ##step2_CNN_classifier
CNN_classifier(x_train_combined,x_test,y_train_combined,y_test)


# for label in os.listdir(test_image_path):
#     test_image = f'./my_writing_dataset/test/{label}'
#     Evaluation(test_image)


# num = random.randint(1,10000)
# Evaluation2(x_test[num])
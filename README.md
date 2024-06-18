# Keras 데이터셋 탐구 (2022.7)

## 개요

본 탐구에서는 python의 tensorflow.py를 이용하여 MNIST 데이터셋을 구분하는 모델을 컴파일하고, 이를 HTML/JS의 웹 상에서 실행하였다.

## 이론

### MNIST 데이터셋

MNIST 데이터베이스 (Modified National Institute of Standards and Technology database)는 손으로 쓴 숫자들로 이루어진 대형 데이터베이스이다. 이 데이터베이스는 또한 기계 학습 분야의 트레이닝 및 테스트에 널리 사용된다.

### tensorflow.py & tensorflow.js

tensorflow는 구글에서 개발한 인공지능 학습 라이브러리로, keras와 같이 사용된다. tensorflow.py는 python 상에서 동작하는 라이브러리로, 이 탐구에서는 모델을 생성하는데 사용되었다. 그리고 tensorflow.js는 JavaScript 상에서 동작하는 라이브러리로, 이 탐구에서는 만들어진 모델을 실행하는데 사용되었다.

## 구현

### 1. 모델 생성

우선, 필요한 tensorflow 라이브러리를 모두 불러온다.
```python
import tensorflow as tf
import tensorflowjs as tfjs # 최종적으로 tf.js 모델로 변환하여 저장할 떄 사용
```
그리고, MNIST 데이터셋을 다운로드받아 필요한 형식으로 포맷한다.
```python
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# 컬러그레이딩
train_images = train_images / 255.0
test_images = test_images / 255.0
```
이제, 모델을 컴파일 및 학습한다.
```python
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)), # MNIST 포맷에 맞게
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))
# 과적합을 방지하기 위하여 validation_data 추가
```
마지막으로, 컴파일된 모델을 tensorflow.js 전용 모델로 변환하여 저장하면 모델 생성이 완료된다.
```python
tfjs.converters.save_keras_model(model, '../web/model')
```

### 2. 모델 활용

이제, HTML/JS 웹 페이지를 만들어, 캔버스에 그림을 입력하면 그것이 어떠한 레이블에 알맞을지를 막대그래프로 표현해보려 한다.

시작하기 전, 사용할 모든 CSS 요소를 정의한다.
```CSS
.container {
    display: grid;
    grid-template-columns: repeat(28, 10px);
    grid-template-rows: repeat(28, 10px);
    gap: 1px;
    margin-bottom: 20px;
}

.pixel {
    width: 10px;
    height: 10px;
    background-color: white;
    border: 1px solid #ddd;
}

.pixel.active {
    background-color: black;
}

.button-container {
    text-align: center;
}

.chart-container {
    width: 80%;
    height: 300px;
    margin: 0 auto;
    display: flex;
    justify-content: space-around;
    align-items: flex-end;
    border: 1px solid #ddd;
    padding: 50px;
}

.bar-wrapper {
    display: flex;
    flex-direction: column;
    align-items: center;
}

.bar {
    width: 30px;
    background-color: steelblue;
    text-align: center;
    color: white;
}
```
첫 번째로, 필요한 HTML 요소 태그들을 정의한다.
```HTML
<div class="container" id="pixel-container"></div>
<div class="button-container">
    <button id="save-button">Submit</button>
</div>
<div class="chart-container" id="chart-container"></div>
```
다음으로, tf.js 라이브러리를 불러오고, 픽셀 캔버스 처리 및 판별 & 그래프 그리기를 JavaScript에서 구현한다.
```HTML
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs"></script>
```
```javascript
const container = document.getElementById('pixel-container');
const saveButton = document.getElementById('save-button');

for (let i = 0; i < 784; i++) {
    const pixel = document.createElement('div');
    pixel.classList.add('pixel');
    pixel.addEventListener('click', () => {
        pixel.classList.toggle('active');
    });
    container.appendChild(pixel);
}

var pixelArray = [];

saveButton.addEventListener('click', async() => {
    pixelArray = []

    const pixels = document.querySelectorAll('.pixel');
    for (let row = 0; row < 28; row++) {
        for (let col = 0; col < 28; col++) {
            const index = row * 28 + col;
            const pixel = pixels[index];
            if (pixel.classList.contains('active')) {
                pixelArray.push(255);
            } else {
                pixelArray.push(0);
            }
        }
    }

    console.log(pixelArray);

    const model = await tf.loadLayersModel('./model/model.json');
    const input = tf.tensor(pixelArray).reshape([1, 28, 28]);

    const prediction = model.predict(input);
    const logits = prediction.dataSync();
    const probabilities = tf.softmax(logits).dataSync();

    label = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
    const chartContainer = document.getElementById('chart-container');
    chartContainer.innerHTML = "";

    probabilities.forEach((value, index) => {
        const bar = document.createElement('div');
        bar.classList.add('bar');
        bar.style.height = `${value * 100}%`;
        bar.textContent = `${value * 100}%`;

        chartContainer.appendChild(bar);
    });
});
```

## 해석 & 결론

본 탐구에서 tensorflow 라이브러리를 다루는 방법을 배우고, 이를 실제 웹 페이지에 연동하는 방법까지 알 수 있었다. 또한 개발 과정에서 갖가지 버그들을 수정하면서 python이라는 언어 자체를 조금 더 능숙하게 사용하는 법을 배우는 계기가 되었던 것 같다. 추후에는 CSS 레이아웃을 개선하여 조금 더 실제 애플리케이션처럼 보이도록 만들었으면 한다.

## 소스 코드 링크

https://github.com/jkmanyeEx/Keras-MNIST
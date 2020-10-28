# 쉬운 모델링을 위하여..

-   ML에 대해서 공부한지 얼마 되지 않았으나 몇몇 대회에 참여하면서 배운 코드를 저장하는 저장소입니다.
-   모델링에만 집중했던 과거를 반성하고, 다시 정리해보고자 만들었습니다.
    -   결국 중요한 부분은 feature engineering이더라고요..
-   변수가공에 집중하고 모델링은 가져다가 쓰기만 해도 되도록 목표를 잡았습니다.
-   관심시가 이미지보다는 tabular data에 있습니다.
-   keras로 공부를 시작하였으나, 최근 tf2를 사용할 수 있도록 코드 연습중에 있습니다.
-   생각없이 구현해보던 코드를 정리한 것이라 출처가 생략되었을 수 있습니다.
    -   혹시 관련된 부분을 알려주시면 조치하겠습니다!

# Tabular Data

## Trees

-   lightgbm과 xgboost계열 모델을 주로 사용합니다.
-   bayes-opt를 사용하여 하이퍼파라미터를 튜닝합니다.
-   pystack을 사용하여 stacking까지 구현할 수 있도록 만들었습니다.

## Neural Networks

-   기본적인 신경망 구조입니다.
-   정확한 이름은 모르지만 앙상블방식을 사용하여 feature extraction부분을 하는 방식을 추가하였습니다.
    -   과적합 문제가 발생할 수 있으나 기본적인 신경망보다는 좋은 성능을 보여주곤 하였습니다.

## GAN-ML

-   개인적인 관심사가 담긴 모델구조입니다.
-   많은 다양한 신경망 구조가 GAN의 구조를 이용하여 개선할 수 있다고 생각합니다.
    -   얼핏 보기에는 actor critic 과 같은 구조입니다.
    -   성능향상의 근거는 Pix2Pix의 논문에서 가져왔습니다.
-   deterministic한 모형에서 벗어나서 확률을 계산할 수 있게 되는 장점이 있습니다.(cgan)
    -   굳이 위의 효과가 필요하지 않아도 적용할 수 있습니다.(pix2pix)

-   generator가 원하는 값을 뱉어내도록 디자인합니다.
    -   모형의 수렴 속도와 성능을 위해서 같은 구조의 네트워크를 pre-train시키고 사용하면 효과가 좋습니다.

# TimeSeires

-   시계열 자료에 사용할 수 있습니다.
    -   코드는 음성 및 신호처리와 관련된 대회에 참여하면서 배웠습니다.
-   RNN기반의 모형들은 너무 느려 선호하지 않아 포함시키지 않았습니다.
    -   어떤 논문에서는 CNN으로만 해도 충분히 적합할 수 있다고도 하더군요
-   WaveNet은 음성합성에서 따온 것으로 알고 있습니다.
-   LSTNet은 논문의 원본 코드를 참조하였습니다.

# Image

-   관련 분야에 대한 데이터에 큰 흥미가 없지만 유용할 만한 내용들을 정리하려고 합니다.
-   현재는 resnet과 unet만 구현해 보았습니다.

# Reinforcement Learning

-   이제 막 공부를 시작하려고 하는 분야입니다.
-   기본적인 내용들 위주로만 구현해보려고 합니다.

# ETC

-   분석하면서 유용했던 간단한 utils코드 입니다.
-   feature selection을 GA알고리즘을 통해서 할 수도 있다는 글을 보고 참고하였습니다.

### TODO

-   RL
    -   DDPG
    -   AC
    -   A2C
-   attention
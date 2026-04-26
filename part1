# Mathematics for Machine Learning 정리 - Part I: Mathematical Foundations

원문: Marc Peter Deisenroth, A. Aldo Faisal, Cheng Soon Ong, *Mathematics for Machine Learning*.
PDF 메타데이터 기준 버전: 2024-01-15.

이 파일은 책의 Part I, 즉 1장부터 7장까지의 수학적 기초를 정리한다. Part I의 목표는 머신러닝 알고리즘을 “블랙박스 사용법”이 아니라 수학적 구조로 이해할 수 있게 만드는 것이다. 저자들은 선형대수, 해석기하, 행렬분해, 벡터 미적분, 확률, 최적화를 서로 분리된 과목처럼 다루지 않고, 뒤의 회귀, PCA, GMM, SVM을 이해하기 위한 하나의 언어로 연결한다.

## 전체 관점

머신러닝은 데이터에서 자동으로 유용한 패턴을 추출하는 방법론이다. 이 책은 머신러닝의 핵심 구성요소를 데이터, 모델, 학습으로 본다. 데이터는 관측된 사실이고, 모델은 데이터를 설명하거나 예측하기 위해 세운 수학적 구조이며, 학습은 모델의 성능이 데이터에 의해 개선되도록 파라미터를 조정하는 과정이다.

Part I은 이 세 요소를 다루기 위한 수학 언어를 만든다.

- 선형대수는 데이터를 벡터와 행렬로 표현하고 선형 변환을 이해하게 한다.
- 해석기하는 벡터공간에 거리, 각도, 직교성, projection이라는 기하학적 의미를 부여한다.
- 행렬분해는 복잡한 선형 변환이나 데이터 행렬을 더 해석 가능한 구성요소로 나눈다.
- 벡터 미적분은 목적함수의 변화율을 계산해 학습 문제를 최적화 문제로 만든다.
- 확률은 데이터, 모델, 예측의 불확실성을 표현한다.
- 연속 최적화는 목적함수나 확률모델에서 좋은 파라미터를 찾는 절차를 제공한다.

## 1장. Introduction and Motivation

1장은 책 전체의 문제의식을 설정한다. 머신러닝은 특정 데이터셋 하나에 맞춘 규칙을 사람이 직접 쓰는 것이 아니라, 다양한 데이터셋에 적용 가능한 일반적 방법론을 설계하는 분야다. 여기서 중요한 말은 “automatic”이다. 알고리즘은 데이터가 주어졌을 때 의미 있는 패턴을 자동으로 찾고, 보지 못한 데이터에도 잘 일반화해야 한다.

저자들은 머신러닝의 중심 개념을 세 가지로 둔다.

- 데이터: 관측값, 입력, 출력, 특징 벡터 등 학습의 재료다.
- 모델: 데이터가 생성되었거나 데이터 사이의 관계가 형성되었다고 가정하는 수학적 구조다.
- 학습: 데이터가 반영된 후 모델의 특정 과제 성능이 좋아지는 과정이다.

이 장의 중요한 메시지는 직관을 수학 언어로 바꿔야 한다는 것이다. 예를 들어 “비슷한 데이터는 비슷하게 예측되어야 한다”는 직관은 거리, norm, inner product, kernel 같은 수학 개념으로 표현된다. “좋은 모델을 찾는다”는 말은 손실함수 최소화, likelihood 최대화, posterior 최대화 같은 문제로 바뀐다. “모델이 불확실하다”는 말은 확률분포와 조건부확률로 다뤄진다.

책은 두 가지 읽기 방식을 제안한다. 수학이 익숙하지 않은 독자는 Part I을 순서대로 읽어 기초 언어를 먼저 쌓고, 수학 배경이 있는 독자는 Part II의 머신러닝 문제를 읽다가 필요한 수학 장으로 돌아갈 수 있다. 이 구조 자체가 책의 관점이다. 수학은 머신러닝과 분리된 준비과목이 아니라, 각 알고리즘의 설계와 해석에 직접 쓰이는 도구다.

## 2장. Linear Algebra

2장은 벡터, 행렬, 벡터공간, 선형사상을 다룬다. 머신러닝에서 데이터는 대부분 벡터 또는 행렬로 표현된다. 이미지, 문서, 센서 측정값, 사용자-아이템 평점표 등은 모두 고차원 벡터나 행렬로 정리할 수 있다. 따라서 선형대수는 머신러닝의 기본 표기법이자 계산 언어다.

### 선형방정식과 행렬

선형방정식계는 여러 개의 선형 제약을 동시에 만족하는 미지수를 찾는 문제다. 예를 들어 `Ax = b`는 행렬 `A`, 미지수 벡터 `x`, 관측 또는 목표 벡터 `b`로 이루어진다. 이 표현은 매우 중요하다. 하나의 선형방정식은 초평면을 의미하고, 방정식계의 해는 여러 초평면의 교차점이다. 해가 없을 수도 있고, 하나일 수도 있고, 무한히 많을 수도 있다.

행렬은 단순한 숫자 표가 아니다. 행렬은 선형방정식계의 계수 모음이면서 동시에 선형 변환을 표현하는 객체다. 행렬 곱 `AB`는 두 선형 변환의 합성으로 볼 수 있고, 행렬-벡터 곱 `Ax`는 벡터 `x`를 새로운 공간 또는 같은 공간의 다른 위치로 보내는 변환이다.

행렬의 기본 연산은 뒤의 모든 장에서 계속 쓰인다.

- 전치 `A^T`: 행과 열을 바꾸며 inner product, covariance, normal equation에 등장한다.
- 역행렬 `A^{-1}`: 정방행렬이 가역일 때 `Ax = b`의 해를 `x = A^{-1}b`로 표현하게 한다.
- 단위행렬 `I`: 곱셈의 항등원이다.
- 대칭행렬: `A = A^T`인 행렬로 covariance, Hessian, Gram matrix 등에서 중요하다.
- 정칙/비정칙: 역행렬 존재 여부와 선형독립성, rank와 연결된다.

### Gaussian elimination과 해의 구조

선형방정식계는 Gaussian elimination으로 풀 수 있다. 핵심 아이디어는 행 연산을 통해 augmented matrix를 더 단순한 row echelon form으로 바꾸는 것이다. 행 연산은 해집합을 보존하면서 방정식계를 더 풀기 쉬운 형태로 만든다.

이 과정에서 pivot column, free variable, rank가 드러난다. pivot은 독립적인 제약을 나타내고, free variable은 해가 여러 개일 때 자유롭게 선택할 수 있는 방향을 나타낸다. 따라서 Gaussian elimination은 단순 계산법이 아니라 선형 시스템의 구조를 파악하는 방법이다.

### 벡터공간

책은 벡터를 기하학적 화살표로만 보지 않는다. 벡터는 덧셈과 스칼라곱이 정의되고 벡터공간의 공리를 만족하는 대상이다. 따라서 숫자 튜플뿐 아니라 다항식, 함수, 행렬도 적절한 연산 아래 벡터로 볼 수 있다.

벡터공간의 핵심은 선형결합이다. 벡터들 `x_1, ..., x_k`와 스칼라 `lambda_1, ..., lambda_k`에 대해 `lambda_1 x_1 + ... + lambda_k x_k`는 이 벡터들이 만들어낼 수 있는 모든 방향과 위치의 조합이다. 어떤 벡터 집합의 모든 선형결합이 만드는 집합을 span이라고 한다.

이 개념은 머신러닝에서 매우 중요하다.

- 선형모델은 특징 벡터 또는 basis function의 선형결합으로 예측을 만든다.
- PCA는 데이터가 놓인 고차원 공간에서 중요한 저차원 부분공간을 찾는다.
- SVM은 특징공간에서 선형 초평면을 찾는다.
- 회귀의 최소제곱해는 관측값을 column space에 projection한 결과로 해석된다.

### 선형독립, basis, rank

벡터 집합이 선형독립이라는 것은 어떤 벡터도 나머지 벡터들의 선형결합으로 표현되지 않는다는 뜻이다. 선형종속이면 중복된 정보가 있다. basis는 어떤 벡터공간을 span하면서 동시에 선형독립인 벡터 집합이다. basis가 정해지면 공간의 모든 벡터를 좌표로 표현할 수 있다.

Rank는 행렬이 담고 있는 독립적인 정보의 수다. column rank는 column space의 차원이고, row rank는 row space의 차원이며 둘은 같다. rank는 선형방정식계의 해 존재성, 역행렬 존재성, 데이터 행렬의 intrinsic dimension, 저랭크 근사와 직접 연결된다.

정방행렬 `A`에 대해 다음 조건들은 서로 깊게 연결된다.

- `A`가 full rank다.
- `A`의 열벡터들이 선형독립이다.
- `A`가 가역이다.
- `Ax = b`가 모든 `b`에 대해 유일한 해를 갖는다.
- `Ax = 0`의 해가 `x = 0`뿐이다.

### 선형사상

선형사상은 덧셈과 스칼라곱을 보존하는 함수다. 즉 `Phi(x + y) = Phi(x) + Phi(y)`, `Phi(lambda x) = lambda Phi(x)`를 만족한다. 유한차원 벡터공간에서 선형사상은 행렬로 표현된다.

Basis를 바꾸면 같은 선형사상도 다른 행렬 표현을 갖는다. 이는 좌표계와 실제 기하학적 변환을 구분해야 한다는 점을 보여준다. PCA, 회전, diagonalization, SVD는 모두 적절한 basis에서 데이터를 더 단순하게 표현하려는 시도다.

### affine space

Affine space는 벡터공간을 어떤 점만큼 이동한 구조다. 선형 부분공간은 원점을 지나야 하지만, affine subspace는 반드시 원점을 지날 필요가 없다. 회귀에서 intercept를 포함한 모델, 데이터의 평균을 중심으로 잡는 PCA, 초평면 분류기는 affine structure와 관련된다.

2장의 핵심은 데이터를 벡터공간의 원소로 보고, 모델과 변환을 행렬 및 선형사상으로 보는 것이다. 이 관점이 뒤의 모든 머신러닝 문제의 기본 골격이 된다.

## 3장. Analytic Geometry

3장은 2장의 추상적 벡터공간에 기하학적 의미를 부여한다. 벡터공간만 있으면 덧셈과 스칼라곱은 가능하지만, 길이, 거리, 각도, 직교성은 아직 정의되지 않는다. 이 장은 inner product를 도입해 벡터공간 안에서 “가깝다”, “비슷하다”, “수직이다”, “투영한다”는 말을 수학적으로 정의한다.

### norm과 거리

Norm은 벡터의 길이를 측정하는 함수다. 일반적인 `p`-norm은 다음 형태다.

`||x||_p = (sum_i |x_i|^p)^{1/p}`

특히 `L2` norm은 Euclidean length이고, `L1` norm은 절댓값 합이며, `Linf` norm은 가장 큰 절댓값 성분이다. Norm이 정의되면 두 벡터 사이의 거리는 `d(x, y) = ||x - y||`로 정의할 수 있다.

머신러닝에서 norm은 여러 역할을 한다.

- 손실함수에서 오차 크기를 측정한다.
- 정규화에서 파라미터 크기를 제한한다.
- k-nearest neighbors나 clustering에서 유사도를 판단한다.
- SVM의 margin은 초평면과 점 사이의 거리로 정의된다.

### inner product와 각도

Inner product는 두 벡터의 곱을 스칼라로 보내는 연산이다. 표준 inner product는 `x^T y`다. Inner product는 norm을 유도한다. `||x|| = sqrt(<x, x>)`가 된다. 또한 Cauchy-Schwarz inequality에 의해 두 벡터 사이의 각도를 다음처럼 정의할 수 있다.

`cos theta = <x, y> / (||x|| ||y||)`

이 식은 유사도 측정의 기초다. 두 벡터가 같은 방향이면 cosine similarity가 1에 가깝고, 직교하면 0이다. 문서 벡터, embedding, feature vector 비교에서 이 관점이 자주 쓰인다.

### 직교성과 orthonormal basis

두 벡터의 inner product가 0이면 직교한다. 직교 basis는 서로 수직인 basis이고, orthonormal basis는 여기에 각 벡터의 길이가 1이라는 조건이 추가된다. Orthonormal basis에서는 좌표 계산이 매우 단순해진다. 벡터 `x`의 basis vector `b_i` 방향 성분은 `<b_i, x>`로 바로 계산된다.

이 개념은 Fourier basis, eigenbasis, principal components와 연결된다. 적절한 orthonormal basis를 찾으면 복잡한 데이터를 독립적인 축들로 분해해 이해할 수 있다.

### orthogonal projection

Projection은 어떤 벡터를 부분공간 위의 가장 가까운 벡터로 보내는 연산이다. 부분공간 `U`에 대한 `x`의 orthogonal projection `pi_U(x)`는 `x - pi_U(x)`가 `U`의 모든 벡터와 직교하도록 정해진다. 즉 남은 residual이 부분공간에 수직이다.

Projection은 책 전체에서 매우 중요하다.

- 선형회귀의 least squares는 관측 벡터 `y`를 design matrix의 column space에 projection하는 문제다.
- PCA는 데이터를 저차원 부분공간에 projection했을 때 reconstruction error를 최소화하거나 variance를 최대화한다.
- SVM의 margin도 점과 초평면 사이의 projection/distance로 계산된다.

Basis가 orthonormal이면 projection은 단순히 inner product들의 합으로 계산된다. Basis가 orthonormal이 아니면 Gram matrix가 등장하고, projection matrix는 더 일반적인 형태를 갖는다.

### rotation

Rotation은 길이와 각도를 보존하는 선형 변환이다. 2차원 회전행렬은 `cos theta`, `sin theta`로 구성되며, determinant가 1이고 orthogonal matrix다. Orthogonal matrix `R`은 `R^T R = I`를 만족해 norm과 inner product를 보존한다.

Rotation은 단순한 기하 예제가 아니라 basis change와 matrix decomposition을 이해하는 준비 단계다. PCA에서 principal axes로 좌표계를 회전시키는 관점, SVD에서 input/output 공간의 orthogonal transformation을 사용하는 관점과 연결된다.

3장의 핵심은 벡터공간에 metric structure를 부여하는 것이다. 이 장을 지나면 데이터 간 거리, 모델 오차, 부분공간 근사, margin 같은 머신러닝 개념을 기하학적으로 이해할 수 있다.

## 4장. Matrix Decompositions

4장은 행렬을 요약하고 분해하는 방법을 다룬다. 행렬은 선형 변환이자 데이터 저장 방식이다. 하지만 큰 행렬을 그대로 보면 구조가 잘 보이지 않는다. Matrix decomposition은 행렬을 더 단순하고 해석 가능한 요소들의 곱으로 표현해 구조를 드러낸다.

### determinant와 trace

Determinant는 정방행렬이 공간의 부피를 얼마나 스케일하는지 나타낸다. 2차원에서는 면적, 3차원에서는 부피의 scaling factor로 이해할 수 있다. Determinant가 0이면 행렬은 공간을 낮은 차원으로 납작하게 만들며, 이 경우 행렬은 invertible하지 않다.

Trace는 정방행렬의 대각성분 합이다. Trace는 cyclic property `tr(ABC) = tr(BCA)` 같은 유용한 성질을 가지며, 행렬 미분과 최적화에서 자주 쓰인다. 뒤의 PCA나 회귀에서 목적함수를 trace 형태로 쓰면 미분과 전개가 쉬워진다.

### eigenvalues와 eigenvectors

행렬 `A`에 대해 `Av = lambda v`를 만족하는 nonzero vector `v`를 eigenvector, scalar `lambda`를 eigenvalue라고 한다. Eigenvector는 선형 변환을 받아도 방향이 변하지 않고 크기만 `lambda`배가 되는 특별한 방향이다.

Eigenvalue/eigenvector는 행렬이 공간을 어떻게 늘이고 줄이는지 알려준다. 특히 대칭행렬은 실수 eigenvalue를 갖고, 서로 직교하는 eigenvector들로 diagonalize할 수 있다. Covariance matrix의 eigenvectors는 데이터 분산의 주요 방향을 나타내며, PCA의 principal components가 된다.

### Cholesky decomposition

Cholesky decomposition은 symmetric positive definite matrix `A`를 `A = LL^T` 형태로 분해한다. 여기서 `L`은 lower triangular matrix다. Positive definite matrix는 모든 nonzero `x`에 대해 `x^T A x > 0`을 만족한다.

Cholesky는 수치적으로 효율적인 선형 시스템 풀이, Gaussian distribution의 covariance 처리, sampling 등에 중요하다. 예를 들어 covariance matrix가 `Sigma = LL^T`이면 표준정규분포 샘플을 선형변환해 원하는 covariance를 가진 Gaussian sample을 만들 수 있다.

### Eigendecomposition과 diagonalization

행렬이 충분한 수의 선형독립 eigenvectors를 가지면 `A = P D P^{-1}`로 쓸 수 있다. `D`는 eigenvalue가 대각에 놓인 diagonal matrix이고, `P`는 eigenvector들을 열로 모은 행렬이다. 이 표현은 복잡한 행렬 곱을 basis change, diagonal scaling, inverse basis change로 나눠 이해하게 한다.

대칭행렬의 경우 더 강한 결과가 성립한다. `A = Q Lambda Q^T`로 분해할 수 있고, `Q`는 orthogonal matrix다. 이 spectral theorem은 covariance matrix, kernel matrix, Hessian 분석에서 핵심이다.

### Singular Value Decomposition

SVD는 임의의 행렬 `A`를 `A = U Sigma V^T`로 분해한다. `U`와 `V`는 orthogonal matrix이고, `Sigma`는 singular values가 대각에 놓인 rectangular diagonal matrix다. SVD는 정방행렬이 아니거나 diagonalizable하지 않은 행렬에도 적용되는 매우 일반적인 분해다.

SVD의 기하학적 의미는 다음과 같다.

1. `V^T`가 입력공간을 회전 또는 반사한다.
2. `Sigma`가 축별로 스케일한다.
3. `U`가 출력공간을 다시 회전 또는 반사한다.

Singular values는 행렬이 각 주요 방향을 얼마나 강하게 늘이는지 나타낸다. 큰 singular value에 해당하는 방향은 데이터 또는 변환의 중요한 구조를 담고, 작은 singular value는 상대적으로 덜 중요한 구조나 noise일 수 있다.

### low-rank approximation

SVD의 중요한 응용은 matrix approximation이다. `A`의 singular values를 큰 순서대로 두고 상위 `k`개만 사용하면 rank-`k` approximation을 얻는다. Eckart-Young theorem에 따르면 이 근사는 Frobenius norm 또는 spectral norm 기준으로 최적의 rank-`k` 근사다.

이 아이디어는 PCA와 직접 연결된다. 데이터 행렬을 저랭크로 근사하면 고차원 데이터를 더 적은 차원으로 표현할 수 있다. 추천시스템의 사용자-아이템 행렬, 이미지 압축, latent factor model도 같은 구조를 갖는다.

4장의 핵심은 행렬을 “더 좋은 좌표계”에서 보면 단순해진다는 것이다. Eigenbasis나 singular vector basis를 찾으면 데이터의 주요 방향, 변환의 스케일, 근사의 품질을 이해할 수 있다.

## 5장. Vector Calculus

5장은 머신러닝 목적함수를 최적화하기 위한 미분 도구를 제공한다. 머신러닝에서는 모델 파라미터를 조정해 손실을 줄이거나 likelihood를 키운다. 이때 파라미터가 벡터나 행렬이면 단변수 미분만으로는 부족하다. Gradient, Jacobian, Hessian, matrix derivative가 필요하다.

### 단변수 미분과 다변수 미분

단변수 함수의 derivative는 입력이 조금 변할 때 출력이 얼마나 변하는지 나타낸다. 다변수 함수 `f: R^D -> R`에서는 각 입력 차원에 대한 partial derivative를 모아 gradient를 만든다.

`grad f(x) = [partial f / partial x_1, ..., partial f / partial x_D]^T`

Gradient는 함수가 가장 빠르게 증가하는 방향을 가리킨다. 따라서 gradient descent는 `x <- x - gamma grad f(x)`처럼 gradient의 반대방향으로 이동해 함수를 줄인다.

### vector-valued function과 Jacobian

함수 `f: R^N -> R^M`처럼 출력도 벡터이면 모든 출력 성분을 모든 입력 성분에 대해 미분한 Jacobian matrix가 필요하다. Jacobian은 국소적으로 함수가 입력공간의 작은 변화를 출력공간에서 어떻게 선형 변환하는지 나타낸다.

Jacobian은 variable transformation, neural network layer, backpropagation, probability density의 change of variables에서 핵심 역할을 한다.

### 행렬에 대한 gradient

머신러닝 파라미터는 종종 행렬이다. 예를 들어 선형변환의 weight matrix, neural network layer, covariance matrix 등이 그렇다. 행렬 미분에서는 scalar-valued function을 matrix에 대해 미분하거나, matrix-valued function을 vector/matrix에 대해 미분한다.

책은 gradient 계산에 유용한 identity를 정리한다. 예를 들어 quadratic form, trace, determinant, inverse 등에 대한 미분 규칙은 회귀, PCA, Gaussian likelihood 최적화에서 반복적으로 쓰인다. Trace trick은 scalar objective를 미분하기 쉬운 형태로 바꾸는 데 특히 유용하다.

### Chain rule과 backpropagation

복합함수의 미분은 chain rule로 계산된다. Neural network는 여러 함수의 합성이다. 각 layer는 affine transformation과 nonlinear activation으로 구성되고, 전체 출력은 이 layer들의 합성이다. Backpropagation은 chain rule을 체계적으로 적용해 최종 손실의 gradient를 모든 중간 파라미터에 효율적으로 전달하는 방법이다.

이 장에서 backpropagation은 딥러닝 전체를 깊게 설명하기보다, vector calculus와 chain rule이 실제 머신러닝 학습 알고리즘으로 이어지는 대표 사례로 제시된다.

### higher-order derivative와 Hessian

Gradient가 1차 변화율이라면 Hessian은 2차 변화율이다. Scalar function `f: R^D -> R`의 Hessian은 모든 second partial derivatives를 모은 `D x D` 행렬이다. Hessian은 함수의 곡률을 나타낸다.

- Hessian이 positive definite이면 국소적으로 볼록하고 local minimum 후보가 된다.
- Hessian이 negative definite이면 local maximum 후보가 된다.
- Hessian이 indefinite이면 saddle point일 수 있다.

Newton method 같은 최적화 알고리즘은 Hessian을 사용해 gradient descent보다 더 곡률 정보를 반영한 업데이트를 한다. 다만 고차원 머신러닝에서는 Hessian 계산과 저장이 비싸므로 gradient 기반 방법이 더 흔하다.

### Taylor approximation과 linearization

Taylor series는 함수를 한 점 근처에서 다항식으로 근사한다. 1차 Taylor approximation은 함수의 local linearization이고, 2차 approximation은 Hessian을 포함해 곡률까지 반영한다.

`f(x + delta) approx f(x) + grad f(x)^T delta`

`f(x + delta) approx f(x) + grad f(x)^T delta + 1/2 delta^T H delta`

이 근사는 최적화 알고리즘의 이론적 배경이다. Gradient descent는 1차 정보를 사용하고, Newton-type method는 2차 정보를 사용한다. 또한 uncertainty propagation이나 local sensitivity 분석에도 쓰인다.

5장의 핵심은 “학습”을 미분 가능한 목적함수의 최적화로 바꾸는 것이다. 이 장의 도구가 있어야 Part II의 maximum likelihood, least squares, PCA objective, EM algorithm, SVM optimization을 계산할 수 있다.

## 6장. Probability and Distributions

6장은 불확실성을 표현하는 수학 언어를 제공한다. 머신러닝에서 불확실성은 여러 곳에서 등장한다. 데이터는 noise를 포함하고, 모델 파라미터는 확실히 알 수 없으며, 예측도 하나의 값이 아니라 분포로 표현될 수 있다.

### probability space

확률공간은 세 요소로 구성된다.

- sample space `Omega`: 가능한 모든 outcome의 집합.
- event space: 관심 있는 outcome들의 부분집합.
- probability measure `P`: event에 확률을 부여하는 함수.

Random variable은 sample space의 outcome을 숫자나 벡터 같은 값으로 매핑하는 함수다. 우리가 실제로 다루는 것은 outcome 자체보다 random variable의 distribution인 경우가 많다.

### discrete와 continuous probability

Discrete random variable은 가능한 값이 셀 수 있는 경우다. Probability mass function `p(x)`는 각 값에 확률을 부여하며, 전체 합은 1이다.

Continuous random variable은 probability density function `f(x)`로 표현된다. 특정 점의 확률은 보통 0이고, 구간 확률은 density를 적분해 얻는다.

이 차이는 중요하다. Discrete case에서는 `P(X = x)`가 의미 있지만, continuous case에서는 `P(X = x) = 0`이고 `P(a <= X <= b)`가 의미 있다.

### sum rule, product rule, Bayes theorem

Joint probability `p(x, y)`는 두 변수가 동시에 특정 값을 가질 확률 또는 밀도다. Marginalization은 joint distribution에서 관심 없는 변수를 합하거나 적분해 제거하는 과정이다.

`p(x) = sum_y p(x, y)` 또는 `p(x) = integral p(x, y) dy`

Product rule은 joint distribution을 conditional distribution과 marginal distribution의 곱으로 쓴다.

`p(x, y) = p(x | y) p(y)`

Bayes theorem은 조건부확률을 뒤집는 규칙이다.

`p(y | x) = p(x | y) p(y) / p(x)`

이 식은 Bayesian inference의 핵심이다. `p(y)`는 prior, `p(x | y)`는 likelihood, `p(y | x)`는 posterior, `p(x)`는 evidence 또는 marginal likelihood로 해석된다.

### summary statistics와 independence

Distribution 전체를 항상 다루기는 어렵기 때문에 mean, variance, covariance 같은 요약통계를 사용한다. Mean은 중심 경향, variance는 퍼짐, covariance는 두 변수의 선형 관계를 나타낸다. Covariance matrix는 다변량 데이터의 방향별 분산과 변수 간 상관을 담는다.

Independence는 `p(x, y) = p(x)p(y)`로 정의된다. Conditional independence는 어떤 변수 `z`가 주어졌을 때 `x`와 `y`가 독립이라는 뜻이다. 이 개념은 probabilistic graphical model에서 매우 중요하다.

### Gaussian distribution

Gaussian distribution은 머신러닝에서 가장 중요한 분포 중 하나다. 단변량 Gaussian은 mean `mu`와 variance `sigma^2`로 정해지고, 다변량 Gaussian은 mean vector `mu`와 covariance matrix `Sigma`로 정해진다.

다변량 Gaussian의 density는 다음 구조를 갖는다.

`N(x | mu, Sigma) proportional |Sigma|^{-1/2} exp(-1/2 (x - mu)^T Sigma^{-1} (x - mu))`

여기서 `(x - mu)^T Sigma^{-1} (x - mu)`는 Mahalanobis distance다. Covariance matrix는 등밀도 곡선의 방향과 폭을 결정한다. Eigenvectors는 ellipsoid의 주축 방향, eigenvalues는 각 방향의 분산을 나타낸다.

Gaussian은 중요한 성질이 많다.

- 선형변환을 해도 Gaussian이다.
- Gaussian 변수들의 marginal과 conditional도 Gaussian이다.
- 많은 noise model에서 자연스럽게 등장한다.
- Least squares와 maximum likelihood를 연결한다.
- Gaussian mixture model의 구성요소가 된다.

### conjugacy와 exponential family

Bayesian inference에서 prior와 likelihood를 결합했을 때 posterior가 prior와 같은 family에 속하면 conjugate prior라고 한다. Conjugacy는 posterior 계산을 단순하게 만든다. 예를 들어 Gaussian likelihood와 적절한 Gaussian prior 조합은 posterior도 Gaussian이 되어 Bayesian linear regression에서 유용하다.

Exponential family는 많은 분포를 하나의 일반 형식으로 묶는다. Gaussian, Bernoulli, multinomial, Poisson 등 여러 분포가 exponential family에 속한다. 이 관점은 통계적 모델링과 generalized linear model, conjugacy 이해에 중요하다.

### change of variables와 inverse transform

Random variable을 함수로 변환하면 distribution도 변한다. Continuous case에서는 단순히 함수를 대입하는 것만으로는 부족하고 Jacobian determinant가 필요하다. 이는 공간의 부피가 변환에 의해 얼마나 늘거나 줄어드는지를 보정한다.

Inverse transform sampling은 uniform random variable을 원하는 distribution의 inverse CDF로 변환해 샘플을 생성하는 방법이다. 이 장의 change-of-variables 관점은 normalizing flows 같은 현대적 모델에도 이어지는 핵심 아이디어다.

6장의 핵심은 머신러닝 모델을 불확실성의 언어로 표현하는 것이다. Part II에서 maximum likelihood, Bayesian regression, GMM, model selection, graphical model을 이해하는 기반이 된다.

## 7장. Continuous Optimization

7장은 학습 문제를 실제로 풀기 위한 최적화 방법을 다룬다. 머신러닝에서 좋은 모델을 찾는 일은 보통 파라미터 `theta`에 대한 objective function을 최소화하거나 최대화하는 문제다. 이 장은 differentiable objective를 전제로 gradient 기반 최적화와 constrained optimization을 소개한다.

### optimization problem의 기본 구조

일반적인 최적화 문제는 다음처럼 쓸 수 있다.

`min_x f(x)`

여기서 `f`는 objective function 또는 loss function이다. 머신러닝에서는 관습적으로 최소화 문제로 쓰는 경우가 많다. Maximum likelihood처럼 최대화 문제가 나오면 negative log-likelihood를 최소화하는 문제로 바꿀 수 있다.

최적화는 크게 unconstrained와 constrained로 나뉜다.

- Unconstrained optimization: 변수 `x`가 전체 공간에서 자유롭게 움직인다.
- Constrained optimization: equality 또는 inequality constraint를 만족해야 한다.

### gradient descent

Gradient descent는 가장 기본적인 1차 최적화 방법이다. Gradient가 함수가 가장 빠르게 증가하는 방향이므로, 그 반대방향으로 이동하면 objective가 줄어들 가능성이 크다.

`x_{k+1} = x_k - gamma_k grad f(x_k)`

여기서 `gamma_k`는 step size 또는 learning rate다. Learning rate가 너무 크면 발산하거나 진동할 수 있고, 너무 작으면 수렴이 매우 느리다. Gradient descent의 핵심은 gradient 계산과 step size 선택이다.

머신러닝에서는 full-batch gradient descent 외에도 stochastic gradient descent와 mini-batch 방법이 중요하지만, 이 책의 이 장은 기본 원리를 이해하는 데 집중한다.

### constrained optimization과 Lagrange multipliers

Constraint가 있는 문제에서는 단순히 gradient가 0인 점을 찾는 것만으로 충분하지 않다. Equality constraint `g(x) = 0` 아래에서 `f(x)`를 최적화하려면 Lagrangian을 만든다.

`L(x, lambda) = f(x) + lambda g(x)`

최적점에서는 objective의 gradient와 constraint의 gradient가 선형종속이 된다. 즉 objective를 더 줄이려는 방향이 constraint surface를 벗어나지 않는 허용 방향과 맞지 않게 된다. Lagrange multiplier는 constraint가 objective에 미치는 민감도를 나타내는 값으로 해석할 수 있다.

이 도구는 SVM의 primal-dual formulation, constrained maximum margin problem, PCA의 unit-norm eigenvector constraint에서 중요하다.

### convex optimization

Convex set은 두 점을 잇는 선분 전체가 집합 안에 남는 집합이다. Convex function은 두 점 사이의 함수값이 chord 아래에 있는 함수다. 직관적으로 그릇 모양이며 local minimum이 global minimum이 된다.

Convex optimization이 중요한 이유는 다음과 같다.

- Local optimum이 global optimum이므로 해석과 알고리즘 설계가 안정적이다.
- 많은 머신러닝 문제, 예를 들어 least squares, logistic regression의 일부 형태, soft-margin SVM 등이 convex structure를 가진다.
- Duality theory를 통해 primal problem을 다른 형태로 바꿔 풀 수 있다.

Convex differentiable function에서는 1차 조건으로 global optimality를 판단할 수 있다. Hessian이 positive semidefinite이면 twice differentiable function이 convex임을 보일 수 있다.

7장의 핵심은 “학습”을 계산 가능한 최적화 절차로 만드는 것이다. Part II에서는 이 장의 도구가 empirical risk minimization, parameter estimation, PCA objective, EM algorithm의 M-step, SVM optimization으로 이어진다.

## Part I 핵심 연결 지도

Part I의 장들은 순서대로 쌓인다.

1. 2장은 데이터와 모델을 벡터/행렬/선형사상으로 표현한다.
2. 3장은 그 공간에 거리, 각도, projection을 부여한다.
3. 4장은 행렬과 데이터의 숨은 구조를 eigenvalue, SVD, low-rank approximation으로 드러낸다.
4. 5장은 objective function을 미분하고 gradient를 계산하는 법을 제공한다.
5. 6장은 noise, uncertainty, inference를 확률분포로 표현한다.
6. 7장은 이 모든 표현을 바탕으로 좋은 파라미터를 찾는 최적화 절차를 제공한다.

이 수학 도구들은 Part II에서 다음처럼 사용된다.

- Linear regression: 선형대수, projection, Gaussian likelihood, gradient/optimization.
- PCA: covariance, eigenvectors, SVD, projection, constrained optimization.
- GMM: Gaussian distribution, latent variables, likelihood, EM optimization.
- SVM: inner product, hyperplane geometry, constrained convex optimization, duality, kernels.

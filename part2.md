# Mathematics for Machine Learning 정리 - Part II: Central Machine Learning Problems

원문: Marc Peter Deisenroth, A. Aldo Faisal, Cheng Soon Ong, *Mathematics for Machine Learning*.
PDF 메타데이터 기준 버전: 2024-01-15.

이 파일은 책의 Part II, 즉 8장부터 12장까지를 정리한다. Part II의 목적은 Part I에서 배운 수학 도구가 실제 머신러닝 문제를 어떻게 구성하고 해결하는지 보여주는 것이다. 저자들은 네 가지 중심 문제를 다룬다.

- Regression: 연속값을 예측한다.
- Dimensionality reduction: 고차원 데이터를 낮은 차원으로 표현한다.
- Density estimation: 데이터가 나온 확률분포를 추정한다.
- Classification: 입력을 이산 class로 분류한다.

Part II는 최신 복잡한 알고리즘을 많이 나열하지 않는다. 대신 대표 알고리즘을 통해 수학적 구조를 명확히 보여준다. 선형회귀는 projection과 Gaussian likelihood, PCA는 eigenvalue/SVD와 low-rank approximation, GMM은 latent variable과 EM, SVM은 margin geometry와 constrained optimization을 보여주는 사례다.

## 8장. When Models Meet Data

8장은 Part I의 수학과 Part II의 머신러닝 문제를 연결하는 다리다. 책은 머신러닝 시스템을 데이터, 모델, 학습의 세 요소로 설명한다. 이 장에서는 “좋은 모델”이 무엇인지, 데이터로부터 파라미터를 어떻게 추정하는지, 확률모델과 최적화가 어떤 관계를 갖는지 정리한다.

### 데이터, 모델, 학습

데이터는 보통 입력 `x_n`과 출력 `y_n`의 쌍 또는 입력만 있는 집합으로 주어진다. 입력은 feature vector로 표현된다. 모델은 데이터의 구조를 설명하는 함수 또는 확률분포다. 학습은 주어진 데이터에 대해 모델의 파라미터를 선택하는 과정이다.

Supervised learning에서는 입력과 정답 출력이 함께 주어진다. Regression은 출력이 연속값이고, classification은 출력이 이산 label이다. Unsupervised learning에서는 출력 label 없이 입력 데이터만 주어지며, 데이터의 구조나 분포를 찾는다. PCA와 GMM은 이런 관점에서 이해할 수 있다.

### empirical risk minimization

좋은 모델을 고르려면 성능을 측정하는 기준이 필요하다. Loss function `l(y, f(x))`는 예측과 정답의 차이를 측정한다. True risk는 실제 데이터 생성분포에 대한 기대손실이지만, 실제 분포를 모르므로 training data에 대한 평균손실인 empirical risk를 사용한다.

`R_emp(f) = 1/N sum_{n=1}^N l(y_n, f(x_n))`

Empirical risk minimization은 이 값을 최소화하는 모델을 찾는 원리다. 하지만 training error만 작게 만들면 overfitting이 발생할 수 있다. 따라서 모델 복잡도, regularization, validation, model selection이 필요하다.

이 장은 “학습”을 최적화 문제로 바꾸는 일반적 틀을 제공한다. Part I의 최적화와 미분이 여기서 사용된다.

### parameter estimation

모델이 파라미터 `theta`를 가진다고 하자. Parameter estimation은 데이터에 가장 잘 맞는 `theta`를 찾는 문제다. 책은 세 가지 중요한 관점을 다룬다.

Maximum likelihood estimation은 데이터가 주어졌을 때 likelihood `p(Y | X, theta)`를 최대화하는 `theta`를 찾는다. 실제 계산에서는 log-likelihood를 사용하며, 곱을 합으로 바꾸어 계산을 안정적이고 단순하게 만든다.

Maximum a posteriori estimation은 prior `p(theta)`를 추가해 posterior `p(theta | data)`를 최대화한다. Bayes theorem에 의해 posterior는 likelihood와 prior의 곱에 비례한다. MAP은 MLE에 regularization을 더한 형태로 해석할 수 있다.

Bayesian inference는 하나의 최적 파라미터만 고르는 대신 posterior distribution 전체를 유지한다. 이는 파라미터 불확실성을 예측 불확실성에 반영할 수 있게 한다.

### probabilistic modeling과 inference

확률모델은 데이터 생성 과정을 확률분포로 표현한다. Observed variable과 latent variable을 구분할 수 있다. Latent variable은 직접 관측되지 않지만 데이터 구조를 설명하기 위해 도입된다. GMM의 cluster assignment, PCA의 latent low-dimensional coordinate가 대표 예다.

Inference는 관측된 데이터로부터 보이지 않는 quantity를 추론하는 과정이다. 예를 들어 latent variable의 posterior를 구하거나, 새로운 입력의 predictive distribution을 계산하는 일이 inference다.

Bayesian predictive distribution은 파라미터의 posterior uncertainty를 적분한다.

`p(y_* | x_*, data) = integral p(y_* | x_*, theta) p(theta | data) dtheta`

이 식은 point estimate보다 풍부한 예측을 제공하지만 계산이 어려울 수 있다.

### directed graphical models

Graphical model은 확률변수 사이의 의존관계를 그래프로 표현한다. Directed graphical model, 즉 Bayesian network에서는 node가 random variable이고 edge가 조건부 의존성을 나타낸다. Joint distribution은 graph 구조에 따라 조건부분포의 곱으로 factorization된다.

이 관점은 복잡한 확률모델을 모듈로 나누어 이해하게 한다. 또한 conditional independence를 시각적으로 파악할 수 있게 한다. GMM이나 Bayesian linear regression도 graphical model로 표현할 수 있다.

### model selection

여러 모델 중 어떤 모델이 좋은지 선택하는 문제는 단순 training error 비교로 해결되지 않는다. 복잡한 모델은 training data에 잘 맞지만 generalization이 나쁠 수 있다. Model selection은 모델 복잡도와 데이터 적합도의 균형을 찾는다.

주요 방법은 다음과 같다.

- validation set 또는 cross-validation으로 generalization 성능을 추정한다.
- regularization으로 복잡한 파라미터를 penalize한다.
- Bayesian evidence 또는 marginal likelihood를 사용해 모델을 비교한다.

8장의 핵심은 머신러닝 알고리즘을 공통 형식으로 보는 것이다. 데이터, 모델, loss/likelihood, optimization/inference, generalization이라는 틀이 뒤의 네 알고리즘에 반복해서 등장한다.

## 9장. Linear Regression

9장은 회귀 문제를 선형모델로 해결한다. Regression은 입력 `x`에서 연속 출력 `y`를 예측하는 문제다. 관측값은 보통 noise를 포함한다고 가정한다.

`y_n = f(x_n) + epsilon_n`

책은 noise를 zero-mean Gaussian으로 두고, 선형모델의 파라미터를 추정하는 여러 관점을 보여준다.

### 문제 설정

입력 `x_n in R^D`, 출력 `y_n in R`가 주어졌을 때 목표는 새로운 입력 `x_*`에 대한 출력 `y_*`를 잘 예측하는 함수 `f`를 찾는 것이다. 선형회귀에서는 함수가 파라미터 `theta`에 대해 선형이라고 가정한다.

가장 단순한 형태는 다음과 같다.

`f(x) = theta^T x`

Bias 또는 intercept를 포함하려면 입력 벡터에 1을 추가해 같은 형식으로 표현할 수 있다. 더 일반적으로 basis function `phi(x)`를 사용하면 입력 자체에는 비선형인 모델도 파라미터에 대해서는 선형인 모델로 쓸 수 있다.

`f(x) = theta^T phi(x)`

이 관점은 중요하다. “Linear regression”의 linear는 입력에 대한 선형성이 아니라 파라미터에 대한 선형성을 의미할 수 있다.

### least squares와 maximum likelihood

Gaussian noise를 가정하면 관측값의 likelihood는 다음과 같은 구조를 가진다.

`p(y | X, theta, sigma^2) = product_n N(y_n | theta^T x_n, sigma^2)`

Log-likelihood를 최대화하는 것은 squared error를 최소화하는 것과 같다. 따라서 least squares는 Gaussian noise model 아래의 maximum likelihood estimation으로 해석된다.

Design matrix `Phi`와 target vector `y`를 사용하면 objective는 다음과 같다.

`||y - Phi theta||^2`

이를 미분해 0으로 두면 normal equation을 얻는다.

`Phi^T Phi theta = Phi^T y`

`Phi^T Phi`가 invertible이면 해는 다음과 같다.

`theta_ML = (Phi^T Phi)^{-1} Phi^T y`

이 식은 선형대수, 미분, 확률이 만나는 지점이다. 행렬 미분으로 최적조건을 구하고, Gaussian likelihood로 loss의 의미를 해석하며, projection으로 기하학적 의미를 이해한다.

### projection 관점

Least squares는 `y`를 design matrix `Phi`의 column space 위로 orthogonal projection하는 문제다. 예측값 `Phi theta`는 column space 안에 있어야 한다. 관측값 `y`가 그 공간에 정확히 놓이지 않으면, 가장 가까운 점을 찾는다. Residual `y - Phi theta`는 column space에 직교한다.

이 관점은 normal equation `Phi^T (y - Phi theta) = 0`와 일치한다. 즉 residual이 모든 column vector와 직교하므로 더 이상 column space 안에서 오차를 줄일 수 없다.

### overfitting과 regularization

Feature가 많거나 모델이 너무 유연하면 training data에는 잘 맞지만 새로운 데이터에는 잘 일반화하지 못할 수 있다. Regularization은 파라미터 크기에 penalty를 주어 모델 복잡도를 제한한다. 대표적으로 ridge regression은 다음 objective를 최소화한다.

`||y - Phi theta||^2 + lambda ||theta||^2`

해는 다음 형태가 된다.

`theta = (Phi^T Phi + lambda I)^{-1} Phi^T y`

Regularization은 MAP estimation으로도 해석된다. Gaussian prior `theta ~ N(0, alpha^{-1}I)`를 두면 posterior maximization은 squared error에 `||theta||^2` penalty를 더한 것과 같다.

### Bayesian linear regression

Bayesian linear regression은 파라미터 `theta`를 고정된 미지수 하나로 보지 않고 random variable로 본다. Prior와 likelihood를 결합해 posterior `p(theta | X, y)`를 계산한다. Gaussian prior와 Gaussian likelihood를 사용하면 posterior도 Gaussian이다.

Bayesian 관점의 장점은 예측 불확실성을 계산할 수 있다는 점이다. 새로운 입력 `x_*`에 대한 predictive distribution은 파라미터 posterior를 적분해 얻는다. 예측분산은 두 부분으로 구성된다.

- Observation noise로 인한 불확실성.
- 파라미터 posterior uncertainty로 인한 불확실성.

데이터가 많은 영역에서는 파라미터 uncertainty가 작고, 데이터가 적은 영역에서는 커진다. 이는 단순 point prediction보다 더 풍부한 모델 해석을 제공한다.

### maximum likelihood as orthogonal projection

책은 ML estimate를 projection 관점으로 다시 해석한다. Gaussian likelihood 아래에서 negative log-likelihood는 squared Euclidean distance와 같고, 이를 최소화하는 예측벡터는 target vector의 column space projection이다. 따라서 회귀는 확률적 추정 문제인 동시에 기하학적 projection 문제다.

9장의 핵심은 하나의 알고리즘을 여러 수학 언어로 동시에 이해하는 것이다.

- 선형대수: design matrix와 normal equation.
- 해석기하: column space projection.
- 확률: Gaussian likelihood와 Bayesian posterior.
- 최적화: squared loss minimization.

## 10장. Dimensionality Reduction with Principal Component Analysis

10장은 PCA를 통해 차원축소를 다룬다. 고차원 데이터는 저장, 시각화, 해석, 계산이 어렵다. 하지만 실제 데이터는 많은 차원이 중복되거나 상관되어 있어 낮은 차원의 구조를 가질 수 있다. PCA는 이 구조를 선형 부분공간으로 찾는 방법이다.

### 문제 설정

데이터 `x_n in R^D`가 주어졌을 때, PCA는 `M < D`인 저차원 표현을 찾는다. 보통 데이터 평균을 빼서 centered data로 만든다. 목표는 데이터를 가장 잘 설명하는 `M`차원 부분공간을 찾는 것이다.

PCA는 여러 등가 관점으로 설명된다.

- Projection 후 reconstruction error를 최소화한다.
- Projection된 데이터의 variance를 최대화한다.
- Covariance matrix의 top eigenvectors를 찾는다.
- Data matrix의 best low-rank approximation을 찾는다.
- Latent variable model의 특수한 경우로 볼 수 있다.

### maximum variance perspective

1차원 principal component를 찾는다고 하자. Unit vector `b` 방향으로 데이터를 projection하면 projection scalar는 `b^T x_n`이다. PCA는 이 projection 값들의 variance가 최대가 되는 방향을 찾는다.

Constraint `||b|| = 1` 아래에서 variance를 최대화하면 covariance matrix의 largest eigenvalue에 해당하는 eigenvector가 해가 된다. 여러 component를 찾을 때는 가장 큰 eigenvalues에 대응하는 eigenvectors를 순서대로 선택한다.

이 관점에서 principal component는 데이터가 가장 많이 퍼져 있는 방향이다. 큰 variance는 데이터의 중요한 변동 구조를 담는다고 해석한다.

### projection perspective

다른 관점에서는 데이터를 저차원 부분공간에 projection한 뒤 다시 원래 공간으로 복원했을 때 reconstruction error가 최소가 되도록 부분공간을 고른다.

`min sum_n ||x_n - projection_U(x_n)||^2`

이 문제의 해 역시 covariance matrix의 top eigenvectors가 span하는 부분공간이다. 즉 variance maximization과 reconstruction error minimization은 같은 PCA 해를 준다.

이 equivalence가 중요하다. PCA는 데이터를 많이 흩어지게 보이는 축을 찾는 방법이면서 동시에 정보를 가장 적게 잃는 선형 압축 방법이다.

### eigenvector computation과 SVD

PCA는 covariance matrix `S`의 eigendecomposition으로 계산할 수 있다. `S = 1/N X^T X`라면 `S`의 top eigenvectors가 principal axes다. 하지만 고차원에서는 covariance matrix를 직접 만들고 분해하는 것이 비쌀 수 있다.

Data matrix `X`의 SVD를 사용하면 PCA를 더 안정적이고 효율적으로 계산할 수 있다.

`X = U Sigma V^T`

여기서 `V`의 column vectors는 principal directions이고, singular values는 각 방향의 variance와 연결된다. 상위 `M`개의 singular vectors를 사용하면 best rank-`M` approximation을 얻는다.

### PCA in high dimensions

데이터 차원 `D`가 샘플 수 `N`보다 훨씬 크면 covariance matrix는 rank가 최대 `N` 또는 `N-1`에 제한된다. 이때 모든 방향을 직접 계산하는 것은 비효율적이다. Dual PCA나 SVD 기반 계산이 유용하다.

고차원 이미지, 유전자 데이터, 텍스트 데이터에서는 이런 문제가 자주 발생한다. PCA는 차원을 줄여 시각화, noise reduction, preprocessing, compression에 사용된다.

### PCA in practice

실제 PCA 절차는 보통 다음과 같다.

1. 데이터 행렬을 구성한다.
2. 각 feature의 평균을 빼서 centering한다.
3. 필요하면 scaling 또는 standardization을 한다.
4. Covariance matrix 또는 SVD를 계산한다.
5. 가장 큰 eigenvalue/singular value에 해당하는 component를 선택한다.
6. 데이터를 선택된 component 공간으로 projection한다.
7. Explained variance ratio를 보고 차원 수를 정한다.

Centering은 매우 중요하다. 평균을 빼지 않으면 첫 component가 데이터의 분산 방향이 아니라 원점에서 평균까지의 방향을 반영할 수 있다.

### latent variable perspective

PCA는 latent variable model로도 볼 수 있다. 관측 데이터 `x`가 낮은 차원 latent variable `z`의 선형변환과 Gaussian noise로 생성된다고 가정한다.

`x = Bz + mu + epsilon`

여기서 `z`는 낮은 차원 latent coordinate이고, `B`는 loading matrix이며, `epsilon`은 Gaussian noise다. 이 관점은 probabilistic PCA로 이어진다. PCA를 단순 선형대수 알고리즘이 아니라 데이터 생성모델로 이해하게 해준다.

10장의 핵심은 고차원 데이터를 가장 중요한 선형 구조로 압축하는 것이다. Part I의 projection, covariance, eigenvectors, SVD, optimization이 모두 사용된다.

## 11장. Density Estimation with Gaussian Mixture Models

11장은 density estimation을 Gaussian mixture model로 다룬다. Density estimation은 데이터가 어떤 확률분포에서 나왔다고 보고 그 분포를 추정하는 문제다. 단일 Gaussian은 하나의 타원형 cluster만 표현할 수 있지만, 실제 데이터는 여러 cluster나 복잡한 형태를 가질 수 있다. GMM은 여러 Gaussian을 섞어 더 유연한 density를 만든다.

### Gaussian mixture model

GMM의 density는 여러 Gaussian component의 weighted sum이다.

`p(x) = sum_{k=1}^K pi_k N(x | mu_k, Sigma_k)`

여기서 `pi_k`는 mixture weight이며 `pi_k >= 0`, `sum_k pi_k = 1`을 만족한다. 각 component는 mean `mu_k`와 covariance `Sigma_k`를 가진다.

GMM은 latent variable `z`를 도입해 해석할 수 있다. `z`는 어떤 component가 선택되었는지를 나타내는 one-hot vector다.

1. 먼저 component `z`를 categorical distribution에서 뽑는다.
2. 선택된 component의 Gaussian에서 `x`를 생성한다.

이 latent-variable 관점이 EM algorithm의 핵심이다.

### maximum likelihood와 어려움

GMM의 파라미터는 `pi_k`, `mu_k`, `Sigma_k`다. 주어진 데이터에 대한 log-likelihood는 다음 구조를 가진다.

`sum_n log sum_k pi_k N(x_n | mu_k, Sigma_k)`

문제는 log 안에 sum이 있다는 것이다. 단일 Gaussian에서는 closed-form MLE를 쉽게 구할 수 있지만, GMM에서는 component assignment가 관측되지 않아 직접 최적화가 어렵다. 각 데이터가 어떤 component에서 왔는지 알면 계산이 쉬워지지만, 그것이 latent variable이라 알 수 없다.

### EM algorithm

Expectation-Maximization algorithm은 latent variable이 있는 likelihood를 최적화하는 반복 절차다. 현재 파라미터로 latent variable의 posterior를 계산하고, 그 posterior를 사용해 파라미터를 업데이트한다.

E-step에서는 responsibility를 계산한다.

`gamma_{nk} = p(z_k = 1 | x_n, theta)`

이는 데이터 `x_n`이 component `k`에 속할 posterior probability다. Bayes theorem에 의해 mixture weight와 Gaussian density를 사용해 계산된다.

M-step에서는 responsibility를 soft assignment로 사용해 파라미터를 업데이트한다.

- `N_k = sum_n gamma_{nk}`
- `pi_k = N_k / N`
- `mu_k = (1/N_k) sum_n gamma_{nk} x_n`
- `Sigma_k = (1/N_k) sum_n gamma_{nk} (x_n - mu_k)(x_n - mu_k)^T`

EM은 매 반복에서 likelihood를 감소시키지 않는 성질이 있다. 하지만 non-convex 문제이므로 local optimum에 수렴할 수 있고, initialization에 민감하다.

### latent-variable perspective

GMM은 “데이터가 여러 숨은 원인 중 하나에 의해 생성된다”는 모델이다. Latent variable `z`는 cluster identity를 나타낸다. Hard clustering인 k-means와 달리 GMM은 soft clustering을 제공한다. 한 데이터 포인트가 여러 component에 부분적으로 속할 수 있다.

Covariance structure에 따라 GMM은 다양한 cluster 형태를 표현한다.

- Spherical covariance: 각 cluster가 원형 또는 구형이다.
- Diagonal covariance: feature 간 covariance를 무시한다.
- Full covariance: 임의의 타원형 cluster를 표현한다.

Mixture component 수 `K`는 model selection 문제다. 너무 작으면 density를 충분히 표현하지 못하고, 너무 크면 overfitting될 수 있다.

### GMM의 의미와 한계

GMM은 density estimation, clustering, anomaly detection, missing data 처리 등에 사용할 수 있다. 하지만 component 수 선택, local optimum, singular covariance 문제, 고차원에서의 covariance 추정 어려움이 있다. Regularization이나 covariance 제약, multiple initialization이 실무적으로 필요하다.

11장의 핵심은 latent variable이 있는 확률모델을 likelihood 기반으로 학습하는 방법이다. EM algorithm은 GMM에만 국한되지 않고, 관측되지 않은 변수가 있는 많은 모델에서 반복적으로 등장하는 원리다.

## 12장. Classification with Support Vector Machines

12장은 binary classification을 Support Vector Machine으로 다룬다. Classification은 입력을 이산 label로 예측하는 문제다. 이 장에서는 label을 `+1`과 `-1`로 두고, 두 class를 나누는 hyperplane을 찾는다.

### separating hyperplanes

선형 classifier는 다음 형태의 decision function을 사용한다.

`f(x) = sign(w^T x + b)`

`w^T x + b = 0`은 decision boundary인 hyperplane이다. `w`는 hyperplane의 normal vector이고, `b`는 offset이다. 어떤 점이 hyperplane의 어느 쪽에 있는지는 `w^T x + b`의 부호로 결정된다.

Linearly separable data에서는 모든 positive examples가 한쪽에, negative examples가 다른 쪽에 놓이는 hyperplane이 존재한다. 하지만 그런 hyperplane은 여러 개일 수 있다. SVM은 그중 margin이 최대인 hyperplane을 선택한다.

### margin과 support vectors

Margin은 decision boundary와 가장 가까운 training point 사이의 거리다. 큰 margin은 classifier가 데이터에 대해 더 안정적이고 일반화가 좋을 가능성이 있음을 의미한다. SVM은 margin을 최대화하는 문제로 정의된다.

제약을 적절히 scaling하면 hard-margin SVM의 primal problem은 다음과 같다.

`min_w,b 1/2 ||w||^2`

subject to

`y_n(w^T x_n + b) >= 1`

Margin을 최대화하는 것은 `||w||`를 최소화하는 것과 같다. 제약을 active하게 만드는 데이터 포인트들이 support vectors다. 이 점들은 decision boundary를 결정하며, 나머지 멀리 떨어진 점들은 해에 직접 영향을 주지 않는다.

### primal SVM

Primal formulation은 geometric margin maximization을 constrained convex optimization problem으로 쓴다. 목적함수는 convex quadratic이고 제약은 linear inequality다. 따라서 convex optimization 도구를 사용할 수 있다.

Linearly separable하지 않은 데이터에는 slack variables `xi_n`을 도입한다. Soft-margin SVM은 margin violation을 허용하되 penalty를 부과한다.

`min_w,b,xi 1/2 ||w||^2 + C sum_n xi_n`

subject to

`y_n(w^T x_n + b) >= 1 - xi_n`, `xi_n >= 0`

`C`는 margin 크기와 training error penalty 사이의 trade-off를 조절한다. `C`가 크면 misclassification을 강하게 벌하고, 작으면 margin을 더 넓게 유지하려 한다.

### dual SVM

Lagrange multiplier를 사용하면 SVM의 dual problem을 얻을 수 있다. Dual formulation의 중요한 특징은 데이터가 inner product `x_i^T x_j` 형태로만 등장한다는 점이다.

Dual variables `alpha_n`은 각 training example의 중요도를 나타낸다. 대부분의 `alpha_n`은 0이고, 0이 아닌 값은 support vectors에 해당한다. 따라서 SVM의 decision function은 support vectors의 weighted combination으로 표현된다.

`f(x) = sign(sum_n alpha_n y_n x_n^T x + b)`

이 sparse representation은 SVM의 중요한 특징이다. 결정경계는 모든 데이터가 아니라 support vectors에 의해 정해진다.

### kernels

Dual formulation에서 데이터가 inner product로만 등장하기 때문에 kernel trick을 사용할 수 있다. 어떤 feature map `phi(x)`가 있을 때 inner product `phi(x_i)^T phi(x_j)`를 직접 계산하지 않고 kernel function `k(x_i, x_j)`로 계산한다.

`k(x_i, x_j) = <phi(x_i), phi(x_j)>`

이를 통해 원래 입력공간에서는 비선형인 decision boundary를 만들 수 있다. 대표 kernel은 다음과 같다.

- Linear kernel: 원래 공간에서 선형 SVM.
- Polynomial kernel: 다항식 feature interaction을 암묵적으로 사용.
- RBF/Gaussian kernel: 매우 유연한 비선형 boundary를 표현.

Kernel은 “고차원 feature space로 명시적으로 변환하지 않고, 그 공간의 inner product만 계산한다”는 아이디어다. 이는 Part I의 inner product와 geometry가 직접 사용되는 대표 사례다.

### numerical solution

SVM은 convex quadratic programming 문제로 풀 수 있다. 실제 알고리즘은 dual 변수와 KKT condition을 이용한다. 책은 numerical solution을 깊게 다루기보다, SVM이 어떤 최적화 문제로 정식화되고 kernel을 통해 어떻게 확장되는지에 초점을 둔다.

SVM의 해석에서 중요한 요소는 다음과 같다.

- `w`: decision boundary의 normal vector.
- `b`: boundary offset.
- margin: boundary와 가장 가까운 점 사이 거리.
- support vectors: margin을 결정하는 training points.
- `C`: soft-margin trade-off parameter.
- kernel: inner product를 통해 비선형 feature space를 암묵적으로 사용하는 함수.

12장의 핵심은 분류 문제를 기하학과 convex optimization으로 이해하는 것이다. SVM은 inner product, distance to hyperplane, Lagrange multiplier, duality, kernel이라는 Part I의 여러 개념이 한 알고리즘 안에서 결합되는 예다.

## Part II 핵심 연결 지도

Part II의 네 문제는 서로 다른 과제를 다루지만 같은 수학 구조를 공유한다.

### Regression: Linear Regression

- 목표: 연속값 예측.
- 모델: 선형함수 또는 basis function에 대한 선형모델.
- 수학: least squares, projection, Gaussian likelihood, Bayesian inference.
- 핵심 결과: MLE는 least squares와 같고, Bayesian linear regression은 예측 불확실성을 제공한다.

### Dimensionality Reduction: PCA

- 목표: 고차원 데이터를 저차원으로 압축.
- 모델: 선형 부분공간 또는 latent variable model.
- 수학: covariance, eigenvectors, SVD, projection, low-rank approximation.
- 핵심 결과: 최대분산 방향과 최소 reconstruction error 방향은 같은 principal components를 준다.

### Density Estimation: GMM

- 목표: 데이터 분포 추정.
- 모델: 여러 Gaussian component의 mixture.
- 수학: Gaussian distribution, latent variables, maximum likelihood, EM algorithm.
- 핵심 결과: 관측되지 않은 component assignment를 responsibility로 추정하며 파라미터를 반복 갱신한다.

### Classification: SVM

- 목표: 이산 class 예측.
- 모델: maximum-margin separating hyperplane.
- 수학: inner product geometry, distance to hyperplane, constrained convex optimization, duality, kernels.
- 핵심 결과: decision boundary는 support vectors가 결정하고, kernel trick으로 비선형 분류를 구현한다.

## Part II를 관통하는 핵심 메시지

Part II의 알고리즘들은 서로 달라 보이지만 모두 Part I의 수학으로 설명된다.

- 좋은 모델을 찾는 문제는 대개 objective function을 최적화하는 문제다.
- Objective는 loss minimization 또는 likelihood maximization에서 온다.
- 행렬과 벡터 표현은 계산을 가능하게 한다.
- Projection과 inner product는 회귀, PCA, SVM의 기하학적 의미를 제공한다.
- 확률분포는 noise, uncertainty, latent variables를 표현한다.
- Decomposition은 고차원 데이터의 숨은 구조를 드러낸다.
- Duality와 kernels는 원래 문제를 더 계산하기 좋거나 표현력 있는 형태로 바꾼다.

따라서 이 책의 Part II는 알고리즘 목록이 아니라, 수학적 언어가 어떻게 머신러닝 문제를 만들고 푸는지를 보여주는 사례집에 가깝다.

# Mathematics for Machine Learning 필기시험 예상문제

기준 교재: Marc Peter Deisenroth, A. Aldo Faisal, Cheng Soon Ong, *Mathematics for Machine Learning*.

이 문서는 인공지능학과 석사과정 필기시험에서 `Mathematics for Machine Learning`을 주요 범위로 삼는다고 가정하고 만든 예상문제와 해설이다. 사용자가 제공한 컴퓨터공학과 예시처럼, 단순 암기보다 “정의 설명”, “간단한 증명”, “표/계산 완성”, “알고리즘 또는 절차 설명” 유형을 중심으로 구성했다.

## 출제 경향 예상

이 책은 머신러닝 알고리즘을 깊게 구현하는 책이라기보다, 머신러닝을 이해하기 위한 수학 구조를 설명하는 책이다. 따라서 필기시험은 다음 유형이 유력하다.

- 선형대수: span, basis, rank, linear independence, eigenvalue/eigenvector, SVD.
- 해석기하: norm, inner product, orthogonality, projection, rotation, distance to hyperplane.
- 미적분/최적화: gradient, Hessian, Taylor approximation, convexity, Lagrange multiplier.
- 확률: Bayes theorem, Gaussian distribution, covariance, independence, maximum likelihood.
- 머신러닝 응용: linear regression, PCA, GMM/EM, SVM의 목적함수와 유도.

특히 “공식을 쓰시오”보다 “왜 그렇게 되는지 증명하시오”, “주어진 수치로 계산하시오”, “기하학적으로 설명하시오” 형태가 나올 가능성이 높다.

## 1. Linear Algebra 예상문제

### 문제 1. 주어진 벡터들이 span하는 공간의 차원을 구하시오

벡터 `v1 = (1, 2, 1)^T`, `v2 = (2, 4, 2)^T`, `v3 = (1, 0, 1)^T`가 있다. 이 벡터들이 span하는 부분공간의 dimension을 구하고, basis 하나를 제시하시오.

#### 해설

Span의 dimension은 주어진 벡터들 중 선형독립인 벡터의 최대 개수, 즉 rank다. 먼저 `v2 = 2v1`이므로 `v2`는 `v1`과 선형종속이다. `v3`가 `v1`의 scalar multiple인지 확인하면 그렇지 않다. 따라서 `v1`, `v3`는 선형독립이고, span의 dimension은 2다.

가능한 basis는 `{v1, v3}`이다.

#### 공부해야 할 내용

- 선형결합, span, linear independence의 정의.
- 행렬의 column rank와 span dimension의 관계.
- Gaussian elimination으로 pivot column을 찾는 방법.

### 문제 2. 행렬의 rank와 해의 개수 관계를 설명하시오

선형방정식계 `Ax = b`에서 `A in R^{m x n}`라고 하자. `rank(A)`와 augmented matrix `rank([A|b])`의 관계에 따라 해가 없는 경우, 유일한 경우, 무한히 많은 경우를 설명하시오.

#### 해설

Rouche-Capelli 정리에 따라 `Ax = b`가 해를 가지려면 `rank(A) = rank([A|b])`이어야 한다.

- `rank(A) != rank([A|b])`: 모순되는 방정식이 있으므로 해가 없다.
- `rank(A) = rank([A|b]) = n`: 모든 변수가 pivot variable이므로 해가 유일하다.
- `rank(A) = rank([A|b]) < n`: free variable이 존재하므로 해가 무한히 많다.

#### 공부해야 할 내용

- Rank, pivot, free variable.
- Row echelon form과 reduced row echelon form.
- Column space 관점에서 `b`가 `A`의 column space에 속해야 해가 존재한다는 해석.

### 문제 3. 선형사상과 행렬 표현을 설명하시오

함수 `T: R^n -> R^m`이 선형사상이라는 것은 무엇인지 정의하고, 모든 선형사상이 행렬곱 `T(x) = Ax`로 표현될 수 있음을 설명하시오.

#### 해설

선형사상은 모든 `x, y`와 scalar `alpha`에 대해 다음을 만족하는 함수다.

- `T(x + y) = T(x) + T(y)`
- `T(alpha x) = alpha T(x)`

표준기저 `e1, ..., en`에 대해 임의의 `x`는 `x = sum_i x_i e_i`로 쓸 수 있다. 선형성에 의해

`T(x) = T(sum_i x_i e_i) = sum_i x_i T(e_i)`

이다. 따라서 `T(e_i)`를 행렬 `A`의 i번째 column으로 놓으면 `T(x) = Ax`가 된다.

#### 공부해야 할 내용

- 표준기저와 좌표 표현.
- 행렬의 column이 선형사상에서 basis vector의 image라는 해석.
- Basis change와 같은 선형사상의 다른 행렬 표현.

### 문제 4. 고유벡터는 unique한지 설명하고 증명하시오

행렬 `A`에 대해 eigenvalue `lambda`와 eigenvector `v`가 있어 `Av = lambda v`를 만족한다고 하자. 이때 eigenvector `v`는 unique한가? 이유를 설명하시오.

#### 해설

일반적으로 unique하지 않다. `v`가 eigenvector이고 `c != 0`이면

`A(cv) = cAv = c lambda v = lambda(cv)`

이므로 `cv`도 같은 eigenvalue `lambda`에 대한 eigenvector다. 따라서 eigenvector의 길이나 부호는 unique하지 않다. 더 나아가 eigenspace의 dimension이 2 이상이면 같은 eigenvalue에 대해 서로 scalar multiple이 아닌 eigenvector도 무한히 많다.

다만 eigenvalue가 simple하고 normalization 조건 `||v|| = 1`을 추가하면 여전히 부호는 unique하지 않다. 즉 `v`와 `-v`가 모두 가능하다.

#### 공부해야 할 내용

- Eigenvalue/eigenvector 정의.
- Eigenspace `null(A - lambda I)`.
- Algebraic multiplicity와 geometric multiplicity의 차이.

### 문제 5. 대칭행렬의 서로 다른 고윳값에 대응하는 고유벡터가 직교함을 증명하시오

실수 대칭행렬 `A = A^T`가 있고, `Av = lambda v`, `Aw = mu w`, `lambda != mu`라고 하자. `v`와 `w`가 직교함을 증명하시오.

#### 해설

대칭행렬이므로 inner product에서 다음이 성립한다.

`<Av, w> = <v, A w>`

왼쪽은 `<lambda v, w> = lambda <v, w>`이고, 오른쪽은 `<v, mu w> = mu <v, w>`이다. 따라서

`lambda <v, w> = mu <v, w>`

이고,

`(lambda - mu)<v, w> = 0`

이다. `lambda != mu`이므로 `<v, w> = 0`이다. 따라서 `v`와 `w`는 직교한다.

#### 공부해야 할 내용

- 대칭행렬의 spectral theorem.
- Inner product의 성질.
- PCA에서 covariance matrix의 eigenvectors가 orthogonal principal directions가 되는 이유.

## 2. Analytic Geometry 예상문제

### 문제 6. 2D rotation matrix가 벡터의 길이를 보존함을 증명하시오

2차원 회전행렬

`R = [[cos theta, -sin theta], [sin theta, cos theta]]`

에 대해 모든 벡터 `x`에 대해 `||Rx||_2 = ||x||_2`임을 증명하시오.

#### 해설

길이 보존은 `R^T R = I`임을 보이면 된다.

`R^T = [[cos theta, sin theta], [-sin theta, cos theta]]`

이므로 곱하면

`R^T R = [[cos^2 theta + sin^2 theta, 0], [0, sin^2 theta + cos^2 theta]] = I`

이다. 따라서

`||Rx||_2^2 = (Rx)^T(Rx) = x^T R^T R x = x^T x = ||x||_2^2`

이고 양변에 제곱근을 취하면 `||Rx||_2 = ||x||_2`다.

#### 공부해야 할 내용

- Orthogonal matrix의 정의 `Q^TQ = I`.
- Rotation matrix와 inner product 보존.
- Norm 보존과 distance 보존의 관계.

### 문제 7. Orthogonal projection의 공식을 유도하시오

단위벡터 `u`가 span하는 1차원 부분공간 `U = span(u)`가 있다. 임의의 벡터 `x`를 `U`에 orthogonal projection한 결과가 `proj_U(x) = (u^T x)u`임을 유도하시오.

#### 해설

Projection 결과는 `alpha u` 형태여야 한다. Orthogonal projection에서는 residual `x - alpha u`가 부분공간 `U`에 직교해야 한다. 따라서

`u^T(x - alpha u) = 0`

이다. `||u|| = 1`이므로 `u^T u = 1`이고,

`u^T x - alpha = 0`

따라서 `alpha = u^T x`다. 그러므로 projection은

`proj_U(x) = (u^T x)u`

이다.

#### 공부해야 할 내용

- Projection의 residual orthogonality 조건.
- Orthonormal basis에서 projection 계산.
- Linear regression의 residual이 design matrix column space에 직교한다는 사실.

### 문제 8. 점과 초평면 사이의 거리를 유도하시오

초평면 `H = {x | w^T x + b = 0}`와 점 `x0`가 있다. 점 `x0`에서 초평면까지의 거리가

`|w^T x0 + b| / ||w||`

임을 설명하시오.

#### 해설

초평면의 normal vector는 `w`다. 점 `x0`에서 초평면 위의 한 점 `xH`까지의 벡터 `x0 - xH`를 normal 방향으로 projection한 길이가 거리다.

초평면 위의 점 `xH`는 `w^T xH + b = 0`을 만족한다. 따라서

`w^T(x0 - xH) = w^T x0 - w^T xH = w^T x0 + b`

이다. Normal 방향 단위벡터는 `w / ||w||`이므로 signed distance는

`(w^T x0 + b) / ||w||`

이고, 거리는 절댓값을 취해

`|w^T x0 + b| / ||w||`

이다.

#### 공부해야 할 내용

- Hyperplane의 normal vector.
- Projection과 거리의 관계.
- SVM margin 유도.

## 3. Matrix Decomposition 예상문제

### 문제 9. PCA와 eigenvalue decomposition의 관계를 설명하시오

Centered data matrix `X`의 covariance matrix를 `S = (1/N)X^T X`라고 하자. PCA의 첫 번째 principal component가 `S`의 가장 큰 eigenvalue에 대응하는 eigenvector임을 설명하시오.

#### 해설

단위벡터 `u` 방향으로 데이터를 projection하면 projection 값은 `Xu`이고, 그 variance는

`u^T S u`

로 표현된다. PCA의 첫 번째 component는 이 variance를 최대화하는 단위벡터를 찾는다.

문제는 다음과 같다.

`maximize u^T S u subject to u^T u = 1`

Lagrangian을 만들면

`L(u, lambda) = u^T S u - lambda(u^T u - 1)`

이고, `u`에 대해 미분하면

`2Su - 2lambda u = 0`

따라서

`Su = lambda u`

이다. 즉 최적해는 covariance matrix의 eigenvector여야 한다. 목적함수 값은 `u^T S u = lambda`이므로 variance를 최대화하려면 가장 큰 eigenvalue에 대응하는 eigenvector를 선택한다.

#### 공부해야 할 내용

- Covariance matrix.
- Rayleigh quotient.
- Lagrange multiplier.
- PCA의 maximum variance perspective.

### 문제 10. SVD를 이용한 low-rank approximation을 설명하시오

행렬 `A`의 SVD가 `A = U Sigma V^T`라고 하자. Rank-`k` approximation `A_k`를 어떻게 만들며, 왜 PCA와 연결되는지 설명하시오.

#### 해설

Singular value를 큰 순서대로 `sigma_1 >= sigma_2 >= ...`라고 할 때, 상위 `k`개 singular value와 그에 대응하는 singular vectors만 남기면

`A_k = sum_{i=1}^k sigma_i u_i v_i^T`

이다. 이는 rank가 최대 `k`인 행렬이다.

Eckart-Young theorem에 따르면 `A_k`는 Frobenius norm 또는 spectral norm 기준으로 `A`에 가장 가까운 rank-`k` 행렬이다. PCA에서 centered data matrix를 SVD하면 상위 right singular vectors가 principal directions가 된다. 따라서 PCA는 데이터를 가장 잘 보존하는 저차원 선형 부분공간을 찾는 low-rank approximation 문제로 볼 수 있다.

#### 공부해야 할 내용

- SVD의 기하학적 의미: rotation/reflection, scaling, rotation/reflection.
- Singular value와 variance의 관계.
- Low-rank approximation과 reconstruction error.

### 문제 11. Positive definite matrix와 Cholesky decomposition의 관계를 설명하시오

Symmetric positive definite matrix의 정의를 쓰고, 왜 Cholesky decomposition `A = LL^T`가 유용한지 설명하시오.

#### 해설

행렬 `A`가 symmetric positive definite라는 것은 `A = A^T`이고 모든 `x != 0`에 대해

`x^T A x > 0`

을 만족한다는 뜻이다. 이러한 행렬은 모든 eigenvalue가 양수이고, Cholesky decomposition `A = LL^T`를 가진다. 여기서 `L`은 lower triangular matrix다.

Cholesky decomposition은 다음에서 유용하다.

- 선형방정식 `Ax = b`를 효율적으로 푼다.
- Gaussian distribution의 covariance matrix를 다룬다.
- 표준정규 샘플 `z`를 `mu + Lz`로 변환해 covariance `A`를 갖는 Gaussian sample을 만들 수 있다.

#### 공부해야 할 내용

- Positive definite와 eigenvalue의 관계.
- Covariance matrix가 positive semidefinite인 이유.
- Gaussian distribution과 covariance matrix.

## 4. Vector Calculus 예상문제

### 문제 12. Quadratic function의 gradient와 Hessian을 구하시오

`f(x) = 1/2 x^T A x + b^T x + c`라고 하자. `A`가 symmetric matrix일 때 `grad f(x)`와 Hessian `H`를 구하시오.

#### 해설

`A`가 symmetric이면

`grad (1/2 x^T A x) = Ax`

이고,

`grad (b^T x) = b`

이다. 따라서

`grad f(x) = Ax + b`

이다. Hessian은 gradient를 다시 미분한 것이므로

`H = A`

이다.

만약 `A`가 symmetric이 아니면 `grad(x^T A x) = (A + A^T)x`이므로 `1/2 x^T A x`의 gradient는 `1/2(A + A^T)x`다.

#### 공부해야 할 내용

- Vector derivative.
- Quadratic form.
- Hessian과 convexity의 관계.

### 문제 13. Chain rule과 backpropagation의 관계를 설명하시오

함수 `L = l(f(g(x)))`가 있다. Chain rule을 사용해 `dL/dx`의 구조를 설명하고, 이것이 backpropagation과 어떻게 연결되는지 설명하시오.

#### 해설

합성함수의 미분은 각 단계의 local derivative를 곱해 계산한다. Scalar case에서는

`dL/dx = dl/df * df/dg * dg/dx`

이다. Vector-valued function에서는 Jacobian의 곱으로 표현된다.

Neural network는 여러 layer의 합성함수다. Backpropagation은 최종 loss에서 시작해 각 layer의 local derivative를 chain rule로 곱하면서 이전 layer와 파라미터에 대한 gradient를 계산하는 절차다. 핵심은 같은 중간 미분값을 재사용해 모든 파라미터의 gradient를 효율적으로 계산한다는 점이다.

#### 공부해야 할 내용

- Scalar chain rule과 Jacobian chain rule.
- Computational graph.
- Gradient descent에서 gradient가 필요한 이유.

### 문제 14. Taylor approximation을 이용해 gradient descent 방향을 설명하시오

미분 가능한 함수 `f`에 대해 1차 Taylor approximation을 사용하여, 왜 `-grad f(x)` 방향이 함수값을 가장 빠르게 감소시키는 방향인지 설명하시오.

#### 해설

작은 변화 `delta`에 대해

`f(x + delta) approx f(x) + grad f(x)^T delta`

이다. `||delta||`가 고정되어 있다고 하자. 함수값을 가장 많이 줄이려면 inner product `grad f(x)^T delta`를 최소화해야 한다. Cauchy-Schwarz inequality에 의해 이 값은 `delta`가 `-grad f(x)` 방향일 때 최소가 된다.

따라서 gradient descent는

`x_{t+1} = x_t - eta grad f(x_t)`

처럼 gradient의 반대방향으로 이동한다.

#### 공부해야 할 내용

- Taylor expansion.
- Cauchy-Schwarz inequality.
- Gradient의 기하학적 의미.

## 5. Probability and Statistics 예상문제

### 문제 15. Bayes theorem을 유도하고 각 항의 의미를 설명하시오

사건 또는 확률변수 `X`, `Y`에 대해 Bayes theorem을 유도하고, 머신러닝에서 prior, likelihood, posterior, evidence가 무엇을 의미하는지 설명하시오.

#### 해설

Product rule에 의해

`p(x, y) = p(x | y)p(y)`

이고 동시에

`p(x, y) = p(y | x)p(x)`

이다. 두 식을 같게 놓으면

`p(y | x)p(x) = p(x | y)p(y)`

따라서

`p(y | x) = p(x | y)p(y) / p(x)`

이다.

Bayesian inference에서 `y`를 parameter 또는 hypothesis, `x`를 data로 보면 다음과 같다.

- `p(y)`: prior, 데이터를 보기 전 믿음.
- `p(x | y)`: likelihood, hypothesis가 주어졌을 때 data가 관측될 가능도.
- `p(y | x)`: posterior, 데이터를 본 후 갱신된 믿음.
- `p(x)`: evidence, normalization constant이자 model selection에 사용 가능.

#### 공부해야 할 내용

- Joint, marginal, conditional probability.
- Bayesian inference.
- MAP과 MLE의 차이.

### 문제 16. Bernoulli와 Binomial distribution의 평균을 증명하시오

`X_i ~ Bernoulli(p)`가 서로 독립이고 `Y = sum_{i=1}^n X_i`라고 하자. `Y ~ Binomial(n, p)`일 때 `E[Y] = np`임을 증명하시오.

#### 해설

기대값의 선형성을 사용하면 독립성 여부와 관계없이

`E[Y] = E[sum_{i=1}^n X_i] = sum_{i=1}^n E[X_i]`

이다. Bernoulli variable은 1일 확률이 `p`, 0일 확률이 `1-p`이므로

`E[X_i] = 1*p + 0*(1-p) = p`

이다. 따라서

`E[Y] = sum_{i=1}^n p = np`

이다.

#### 공부해야 할 내용

- Bernoulli distribution.
- Binomial distribution.
- 기대값의 선형성.

### 문제 17. 다변량 Gaussian의 covariance matrix가 하는 역할을 설명하시오

다변량 Gaussian `N(mu, Sigma)`에서 `mu`와 `Sigma`의 의미를 설명하고, `Sigma`의 eigenvalue/eigenvector가 density의 모양과 어떻게 연결되는지 설명하시오.

#### 해설

`mu`는 분포의 중심, 즉 평균벡터다. `Sigma`는 covariance matrix로 각 방향의 분산과 변수 간 공분산을 담는다. 다변량 Gaussian의 density는 대략 다음 항을 포함한다.

`exp(-1/2 (x - mu)^T Sigma^{-1}(x - mu))`

여기서 `(x - mu)^T Sigma^{-1}(x - mu)`는 Mahalanobis distance다. `Sigma`가 대칭 positive definite이면 eigenvalue decomposition을 통해 principal axes를 얻을 수 있다.

- Eigenvector: Gaussian ellipsoid의 주축 방향.
- Eigenvalue: 해당 방향의 분산 크기.
- 큰 eigenvalue: 그 방향으로 데이터가 넓게 퍼짐.
- 작은 eigenvalue: 그 방향으로 데이터가 좁게 모임.

#### 공부해야 할 내용

- Covariance matrix.
- Mahalanobis distance.
- Eigen decomposition의 기하학적 해석.

### 문제 18. Maximum likelihood와 MAP의 차이를 설명하시오

Parameter `theta`와 data `D`가 있을 때 MLE와 MAP estimate의 정의를 쓰고, 두 방법의 차이를 설명하시오.

#### 해설

MLE는 likelihood를 최대화한다.

`theta_MLE = argmax_theta p(D | theta)`

MAP은 posterior를 최대화한다.

`theta_MAP = argmax_theta p(theta | D)`

Bayes theorem에 의해

`p(theta | D) proportional p(D | theta)p(theta)`

이므로

`theta_MAP = argmax_theta p(D | theta)p(theta)`

이다. 즉 MAP은 likelihood에 prior를 곱한 것을 최대화한다. Prior가 uniform이면 MAP과 MLE는 같아진다. Gaussian prior를 두면 파라미터 크기를 작게 유지하는 regularization 효과가 생긴다.

#### 공부해야 할 내용

- Likelihood와 probability의 차이.
- Prior/posterior.
- Regularization의 Bayesian 해석.

## 6. Optimization 예상문제

### 문제 19. Convex function인지 판정하시오

함수 `f(x) = x^T A x + b^T x + c`가 있다. `A`가 symmetric matrix일 때, `f`가 convex이기 위한 조건을 쓰고 증명하시오.

#### 해설

Twice differentiable function이 convex이기 위한 충분조건이자 이 경우의 필요조건은 Hessian이 positive semidefinite인 것이다.

`f(x) = x^T A x + b^T x + c`

에서 `A`가 symmetric이면

`grad f(x) = 2Ax + b`

이고

`H = 2A`

이다. 따라서 `f`가 convex이기 위한 조건은 `2A`가 positive semidefinite인 것, 즉 `A`가 positive semidefinite인 것이다.

`A`가 positive semidefinite이면 모든 `z`에 대해 `z^T A z >= 0`이므로 `z^T H z = 2z^T A z >= 0`이다. 따라서 Hessian이 positive semidefinite이고 `f`는 convex다.

#### 공부해야 할 내용

- Convex set과 convex function의 정의.
- Hessian positive semidefinite 조건.
- Quadratic objective와 least squares.

### 문제 20. Lagrange multiplier를 이용해 constrained optimization을 푸시오

`maximize x^T A x subject to x^T x = 1` 문제에서 stationary point가 `A`의 eigenvector임을 보이시오. 단, `A`는 symmetric matrix라고 하자.

#### 해설

Lagrangian을 다음과 같이 둔다.

`L(x, lambda) = x^T A x - lambda(x^T x - 1)`

`x`에 대해 미분하면

`grad_x L = 2Ax - 2lambda x`

이다. Stationary condition은

`2Ax - 2lambda x = 0`

즉

`Ax = lambda x`

이다. 따라서 stationary point `x`는 `A`의 eigenvector이고, objective value는

`x^T A x = lambda x^T x = lambda`

이다. Maximum은 가장 큰 eigenvalue에 대응하는 eigenvector에서 달성된다.

#### 공부해야 할 내용

- Lagrange multiplier.
- Rayleigh quotient.
- PCA maximum variance derivation.

### 문제 21. Gradient descent 알고리즘을 설명하고 learning rate의 영향을 쓰시오

Differentiable objective `f(theta)`를 최소화하는 gradient descent의 update rule을 쓰고, learning rate가 너무 크거나 작을 때 어떤 문제가 생기는지 설명하시오.

#### 해설

Gradient descent는 현재 위치에서 objective가 가장 빠르게 증가하는 방향인 gradient의 반대방향으로 이동한다.

`theta_{t+1} = theta_t - eta grad f(theta_t)`

여기서 `eta`는 learning rate다.

- `eta`가 너무 작으면 수렴이 매우 느리다.
- `eta`가 너무 크면 minimum을 지나쳐 진동하거나 발산할 수 있다.
- 적절한 `eta`는 objective의 곡률, scaling, condition number에 영향을 받는다.

Convex quadratic에서는 learning rate와 Hessian의 eigenvalue가 수렴 속도에 큰 영향을 준다.

#### 공부해야 할 내용

- Gradient의 의미.
- Step size와 convergence.
- Convex optimization의 기본 개념.

## 7. Linear Regression 예상문제

### 문제 22. Least squares 해를 유도하시오

Design matrix `Phi in R^{N x D}`와 target vector `y in R^N`가 있다. Linear regression의 objective

`J(theta) = ||y - Phi theta||^2`

를 최소화하는 `theta`를 유도하시오. 단, `Phi^T Phi`는 invertible이라고 하자.

#### 해설

Objective를 전개하면

`J(theta) = (y - Phi theta)^T(y - Phi theta)`

이다. `theta`에 대해 미분하면

`grad J(theta) = -2Phi^T y + 2Phi^T Phi theta`

이다. 최적점에서 gradient가 0이므로

`Phi^T Phi theta = Phi^T y`

이다. `Phi^T Phi`가 invertible이면

`theta = (Phi^T Phi)^{-1}Phi^T y`

이다.

#### 공부해야 할 내용

- Matrix calculus.
- Normal equation.
- Full column rank 조건.

### 문제 23. Linear regression을 orthogonal projection으로 설명하시오

Least squares solution에서 residual `r = y - Phi theta`가 `Phi`의 column space에 직교함을 보이고, 이를 projection 관점에서 설명하시오.

#### 해설

Normal equation은

`Phi^T Phi theta = Phi^T y`

이고 이를 정리하면

`Phi^T(y - Phi theta) = 0`

이다. 즉

`Phi^T r = 0`

이다. 이는 residual `r`이 `Phi`의 모든 column vector와 직교한다는 뜻이다.

예측값 `Phi theta`는 `Phi`의 column space 안에 있다. Least squares는 `y`를 이 column space 안의 가장 가까운 점으로 projection하는 문제다. Projection 후 남은 residual은 projection된 부분공간에 직교해야 하므로 normal equation과 같은 조건이 나온다.

#### 공부해야 할 내용

- Column space.
- Orthogonal projection.
- Least squares의 기하학적 의미.

### 문제 24. Gaussian noise 가정에서 least squares가 MLE와 같음을 보이시오

관측모델이 `y_n = theta^T x_n + epsilon_n`, `epsilon_n ~ N(0, sigma^2)`라고 하자. Maximum likelihood estimation이 squared error minimization과 같음을 보이시오.

#### 해설

Gaussian noise 가정 아래에서

`p(y_n | x_n, theta) = N(y_n | theta^T x_n, sigma^2)`

이다. 독립 관측이라면 likelihood는 곱이다.

`p(y | X, theta) = product_n N(y_n | theta^T x_n, sigma^2)`

Log-likelihood를 쓰면 상수항을 제외하고

`log p(y | X, theta) = -1/(2sigma^2) sum_n (y_n - theta^T x_n)^2 + const`

이다. 이를 최대화하는 것은

`sum_n (y_n - theta^T x_n)^2`

를 최소화하는 것과 같다. 따라서 Gaussian noise 아래의 MLE는 least squares와 같다.

#### 공부해야 할 내용

- Gaussian likelihood.
- Log-likelihood.
- MLE와 loss minimization의 연결.

## 8. PCA 예상문제

### 문제 25. PCA의 두 관점이 같은 해를 주는 이유를 설명하시오

PCA의 maximum variance 관점과 minimum reconstruction error 관점이 같은 principal components를 준다는 사실을 설명하시오.

#### 해설

Maximum variance 관점에서는 데이터를 어떤 저차원 부분공간에 projection했을 때 projection된 데이터의 variance가 최대가 되도록 한다. 1차원에서는 `u^T S u`를 `||u|| = 1` 조건 아래 최대화하고, 해는 covariance matrix의 largest eigenvalue eigenvector다.

Minimum reconstruction error 관점에서는 데이터를 부분공간에 projection한 뒤 다시 원래 공간으로 복원했을 때 오차

`sum_n ||x_n - projection_U(x_n)||^2`

를 최소화한다. Centered data에서 전체 variance는 고정되어 있고, reconstruction error를 줄이는 것은 projection된 variance를 늘리는 것과 같다. 따라서 두 문제는 같은 eigenvectors를 선택한다.

#### 공부해야 할 내용

- PCA의 maximum variance derivation.
- Orthogonal projection.
- Reconstruction error와 explained variance.

### 문제 26. PCA 계산 절차를 쓰시오

데이터 행렬이 주어졌을 때 PCA를 수행하는 절차를 순서대로 쓰고, 각 단계의 의미를 설명하시오.

#### 해설

일반적인 PCA 절차는 다음과 같다.

1. 데이터 행렬 `X`를 구성한다.
2. 각 feature의 평균을 빼서 centering한다.
3. 필요하면 feature scaling을 수행한다.
4. Covariance matrix `S = (1/N)X^T X`를 계산하거나 `X`의 SVD를 계산한다.
5. 가장 큰 eigenvalue 또는 singular value에 대응하는 방향을 선택한다.
6. 선택한 principal components에 데이터를 projection한다.
7. Explained variance ratio를 보고 차원 수를 결정한다.

Centering을 하지 않으면 principal component가 평균 방향의 영향을 크게 받을 수 있다.

#### 공부해야 할 내용

- Centering의 이유.
- Covariance matrix와 SVD.
- Explained variance ratio.

## 9. GMM과 EM 예상문제

### 문제 27. Gaussian mixture model을 정의하고 각 파라미터의 의미를 설명하시오

GMM의 density를 쓰고, mixture weight, mean, covariance의 의미를 설명하시오.

#### 해설

GMM은 여러 Gaussian component의 weighted sum으로 density를 표현한다.

`p(x) = sum_{k=1}^K pi_k N(x | mu_k, Sigma_k)`

여기서

- `pi_k`: k번째 component가 선택될 확률. `pi_k >= 0`, `sum_k pi_k = 1`.
- `mu_k`: k번째 Gaussian의 중심.
- `Sigma_k`: k번째 Gaussian의 퍼짐과 방향.

Latent variable `z`를 사용하면 먼저 component를 선택하고, 선택된 Gaussian에서 데이터를 생성하는 모델로 볼 수 있다.

#### 공부해야 할 내용

- Mixture model.
- Latent variable.
- Gaussian distribution.

### 문제 28. EM algorithm의 E-step과 M-step을 설명하시오

GMM에서 EM algorithm이 필요한 이유를 설명하고, E-step과 M-step에서 무엇을 계산하는지 쓰시오.

#### 해설

GMM의 log-likelihood는

`sum_n log sum_k pi_k N(x_n | mu_k, Sigma_k)`

처럼 log 안에 sum이 있어 직접 최적화가 어렵다. 이는 각 데이터가 어떤 component에서 생성되었는지 나타내는 latent variable이 관측되지 않기 때문이다.

E-step에서는 현재 파라미터로 responsibility를 계산한다.

`gamma_{nk} = p(z_k = 1 | x_n)`

이는 데이터 `x_n`이 component `k`에 속할 posterior probability다.

M-step에서는 responsibility를 soft assignment로 사용해 파라미터를 갱신한다.

- `N_k = sum_n gamma_{nk}`
- `pi_k = N_k / N`
- `mu_k = (1/N_k) sum_n gamma_{nk} x_n`
- `Sigma_k = (1/N_k) sum_n gamma_{nk}(x_n - mu_k)(x_n - mu_k)^T`

EM은 likelihood를 반복적으로 증가시키지만 non-convex 문제이므로 local optimum에 수렴할 수 있다.

#### 공부해야 할 내용

- Latent variable posterior.
- Responsibility.
- Maximum likelihood with hidden variables.

## 10. SVM 예상문제

### 문제 29. Hard-margin SVM의 primal optimization problem을 유도하시오

Binary label `y_n in {-1, +1}`와 classifier `f(x) = sign(w^T x + b)`가 있다. Hard-margin SVM의 목적함수와 제약식을 쓰고, 왜 margin maximization이 `||w||` minimization으로 바뀌는지 설명하시오.

#### 해설

분류가 올바르려면 모든 `n`에 대해

`y_n(w^T x_n + b) > 0`

이어야 한다. Scale ambiguity를 제거하기 위해 support vector에서 functional margin이 1이 되도록 두면 제약은

`y_n(w^T x_n + b) >= 1`

이다.

초평면과 점 사이의 거리는

`|w^T x_n + b| / ||w||`

이고, support vector에서는 `|w^T x_n + b| = 1`이므로 geometric margin은 `1/||w||`이다. 두 class 사이의 전체 margin은 `2/||w||`에 비례한다. 따라서 margin을 최대화하는 것은 `||w||`를 최소화하는 것과 같다.

Hard-margin SVM은 다음 문제다.

`min_{w,b} 1/2 ||w||^2`

subject to

`y_n(w^T x_n + b) >= 1`

#### 공부해야 할 내용

- Hyperplane과 margin.
- Functional margin과 geometric margin.
- Constrained convex optimization.

### 문제 30. Soft-margin SVM에서 slack variable과 C의 의미를 설명하시오

Linearly separable하지 않은 데이터에서 soft-margin SVM을 사용하는 이유와 slack variable `xi_n`, regularization parameter `C`의 의미를 설명하시오.

#### 해설

현실의 데이터는 완전히 선형분리되지 않는 경우가 많다. Hard-margin SVM은 이 경우 feasible solution이 없을 수 있다. Soft-margin SVM은 slack variable `xi_n >= 0`을 도입해 margin violation을 허용한다.

문제는 다음과 같다.

`min_{w,b,xi} 1/2 ||w||^2 + C sum_n xi_n`

subject to

`y_n(w^T x_n + b) >= 1 - xi_n`

`xi_n = 0`이면 margin 조건을 만족한다. `0 < xi_n < 1`이면 margin 안쪽에 있지만 분류는 맞을 수 있다. `xi_n > 1`이면 오분류일 수 있다.

`C`는 margin을 넓게 유지하려는 목표와 training error를 줄이려는 목표 사이의 trade-off를 조절한다.

#### 공부해야 할 내용

- Hard-margin과 soft-margin의 차이.
- Slack variable.
- Regularization parameter.

### 문제 31. Kernel trick을 설명하시오

SVM의 dual formulation에서 kernel trick이 가능한 이유를 설명하고, linear kernel, polynomial kernel, RBF kernel의 차이를 쓰시오.

#### 해설

SVM의 dual formulation에서는 데이터가 inner product `x_i^T x_j` 형태로만 등장한다. 만약 feature map `phi(x)`를 사용해 고차원 공간에서 선형분류를 한다면 inner product는 `phi(x_i)^T phi(x_j)`가 된다.

Kernel function은 이 값을 직접 feature map을 계산하지 않고 구한다.

`k(x_i, x_j) = <phi(x_i), phi(x_j)>`

따라서 고차원 또는 무한차원 feature space에서의 선형분류를 원래 입력공간에서는 비선형 decision boundary로 구현할 수 있다.

- Linear kernel: `k(x, z) = x^T z`. 원래 공간의 선형분류.
- Polynomial kernel: 다항식 feature interaction을 반영.
- RBF kernel: 거리에 기반한 매우 유연한 비선형 decision boundary.

#### 공부해야 할 내용

- Dual SVM.
- Inner product와 feature map.
- Kernel의 positive semidefinite 조건.

## 11. 종합형 예상문제

### 문제 32. Linear regression, PCA, SVM을 projection 관점에서 비교하시오

세 알고리즘에서 projection 또는 geometry가 어떤 역할을 하는지 비교 설명하시오.

#### 해설

Linear regression에서는 target vector `y`를 design matrix `Phi`의 column space에 orthogonal projection한다. 이때 residual은 column space에 직교한다.

PCA에서는 고차원 데이터를 저차원 principal subspace에 projection한다. 목표는 projection된 variance를 최대화하거나 reconstruction error를 최소화하는 것이다.

SVM에서는 점과 separating hyperplane 사이의 거리를 계산하고, 가장 가까운 점들인 support vectors와 hyperplane 사이의 margin을 최대화한다. 여기서 distance to hyperplane은 normal vector 방향 projection으로 유도된다.

세 알고리즘 모두 inner product, norm, orthogonality가 핵심이다. 다만 regression은 target approximation, PCA는 data representation, SVM은 classification boundary에 초점이 있다.

#### 공부해야 할 내용

- Orthogonal projection.
- Column space.
- Principal subspace.
- Distance to hyperplane.

### 문제 33. MLE 관점에서 linear regression과 GMM을 비교하시오

Linear regression과 GMM에서 maximum likelihood가 어떻게 사용되는지 비교하시오.

#### 해설

Linear regression에서는 Gaussian noise를 가정하면 log-likelihood maximization이 squared error minimization과 같아진다. 이 경우 closed-form solution이 normal equation으로 주어진다.

GMM에서도 likelihood를 최대화하지만, 각 데이터가 어떤 Gaussian component에서 왔는지 모르는 latent variable이 있다. Log-likelihood에 `log sum` 구조가 생겨 closed-form으로 직접 풀기 어렵다. 그래서 EM algorithm을 사용해 latent assignment의 posterior를 추정하고, 그 값을 바탕으로 파라미터를 반복 갱신한다.

즉 linear regression은 MLE가 직접 최적화되어 closed-form solution을 갖는 대표 사례이고, GMM은 latent variable 때문에 iterative MLE가 필요한 사례다.

#### 공부해야 할 내용

- Gaussian likelihood.
- Closed-form MLE.
- Latent variable과 EM.

### 문제 34. PCA와 GMM의 latent variable 관점을 비교하시오

PCA와 GMM을 latent variable model로 볼 때 각각의 latent variable이 무엇을 의미하는지 설명하시오.

#### 해설

PCA의 latent variable은 낮은 차원의 연속적인 coordinate다. 관측 데이터 `x`는 latent variable `z`가 선형변환되고 noise가 더해져 생성된다고 볼 수 있다.

`x = Bz + mu + epsilon`

GMM의 latent variable은 이산적인 component assignment다. 각 데이터가 어떤 Gaussian component에서 생성되었는지를 나타낸다.

따라서 PCA는 continuous latent representation을 통해 dimensionality reduction을 수행하고, GMM은 discrete latent assignment를 통해 clustering과 density estimation을 수행한다.

#### 공부해야 할 내용

- Latent variable.
- Probabilistic PCA의 기본 아이디어.
- Mixture model의 component assignment.

### 문제 35. Regularization을 optimization과 Bayesian 관점에서 설명하시오

Ridge regression의 regularization term `lambda ||theta||^2`를 optimization 관점과 Bayesian 관점에서 설명하시오.

#### 해설

Optimization 관점에서 regularization은 training error뿐 아니라 파라미터 크기도 함께 penalize한다.

`||y - Phi theta||^2 + lambda ||theta||^2`

이렇게 하면 지나치게 큰 파라미터를 피하고, overfitting을 줄이며, `Phi^T Phi`가 singular하거나 ill-conditioned일 때도 더 안정적인 해를 얻는다.

Bayesian 관점에서는 Gaussian prior `theta ~ N(0, alpha^{-1}I)`를 둔 MAP estimation으로 해석할 수 있다. Prior는 파라미터가 0 근처에 있을 가능성이 높다고 가정하므로 큰 파라미터를 억제한다. 따라서 L2 regularization은 Gaussian prior와 대응된다.

#### 공부해야 할 내용

- Ridge regression.
- MAP estimation.
- Gaussian prior와 L2 penalty.

## 12. 실제 시험 대비용 핵심 암기/증명 목록

### 반드시 쓸 수 있어야 하는 정의

- Vector space, subspace, span, basis, rank.
- Linear map, matrix representation.
- Norm, inner product, orthogonality, orthonormal basis.
- Eigenvalue/eigenvector, eigenspace.
- Positive definite, covariance matrix.
- Gradient, Jacobian, Hessian.
- Convex function, convex optimization.
- Probability space, random variable, likelihood, posterior.
- Empirical risk, MLE, MAP.
- PCA, GMM, EM, SVM, kernel.

### 반드시 유도할 수 있어야 하는 식

- `R^T R = I`이면 `||Rx|| = ||x||`.
- Projection formula `proj_u(x) = (u^T x)u` for unit `u`.
- Point-to-hyperplane distance `|w^T x + b| / ||w||`.
- Least squares normal equation `Phi^T Phi theta = Phi^T y`.
- Gaussian noise에서 least squares와 MLE의 동치.
- PCA objective `maximize u^T S u subject to u^T u = 1`.
- PCA stationary condition `Su = lambda u`.
- Bayes theorem `p(theta | D) proportional p(D | theta)p(theta)`.
- GMM responsibility `gamma_{nk}`의 의미.
- Hard-margin SVM primal problem.

### 계산 문제로 연습해야 하는 것

- 주어진 벡터들의 rank와 basis 구하기.
- 주어진 작은 행렬의 eigenvalue/eigenvector 계산.
- 2D projection 계산.
- 작은 데이터셋의 covariance matrix 계산.
- PCA 첫 번째 principal component 찾기.
- Linear regression normal equation 풀기.
- Bernoulli/Binomial 기대값과 분산 계산.
- Hessian으로 convexity 판정.
- SVM margin 계산.

### 서술형으로 대비해야 하는 비교 주제

- MLE vs MAP.
- Regression vs classification.
- PCA maximum variance vs reconstruction error.
- Hard-margin SVM vs soft-margin SVM.
- K-means hard assignment vs GMM soft assignment.
- Gradient descent vs closed-form solution.
- Eigen decomposition vs SVD.
- Frequentist point estimate vs Bayesian posterior.

## 13. 모의시험 세트

아래는 실제 시험처럼 5문제 정도로 압축한 모의 세트다.

### 모의시험 A

1. 주어진 벡터들이 span하는 공간의 dimension을 구하고 basis를 제시하시오.
2. 2D rotation matrix가 벡터의 길이와 inner product를 보존함을 증명하시오.
3. Gaussian noise 가정에서 linear regression의 MLE가 least squares와 같음을 증명하시오.
4. PCA의 첫 번째 principal component가 covariance matrix의 largest eigenvalue eigenvector임을 유도하시오.
5. Hard-margin SVM의 primal optimization problem을 쓰고 margin maximization과의 관계를 설명하시오.

### 모의시험 B

1. 대칭행렬의 서로 다른 eigenvalue에 대응하는 eigenvector들이 orthogonal임을 증명하시오.
2. 함수 `f(x) = x^T A x + b^T x + c`의 convexity 조건을 Hessian으로 판정하시오.
3. Bayes theorem을 유도하고 MLE와 MAP의 차이를 설명하시오.
4. GMM의 EM algorithm에서 E-step과 M-step을 설명하시오.
5. Kernel trick이 무엇이며 SVM에서 왜 가능한지 설명하시오.

### 모의시험 C

1. `Ax = b`의 해가 없거나 유일하거나 무한히 많은 조건을 rank로 설명하시오.
2. Orthogonal projection 공식을 유도하고 least squares와 연결하시오.
3. SVD를 이용한 rank-`k` approximation과 PCA의 관계를 설명하시오.
4. 다변량 Gaussian에서 covariance matrix의 eigenvalue/eigenvector가 density 모양과 어떻게 연결되는지 설명하시오.
5. Linear regression, PCA, SVM을 모두 projection/geometry 관점에서 비교하시오.

## 14. 우선순위별 공부 전략

### 1순위: 증명형 핵심

가장 먼저 준비해야 할 것은 짧은 증명이다. 제공된 기초수학 예시도 rotation matrix, eigenvector uniqueness, convexity처럼 정의에서 출발해 5-10줄로 증명하는 문제가 많다.

우선 다음 증명은 손으로 반복해서 써봐야 한다.

- Rotation matrix의 norm 보존.
- Orthogonal projection 공식.
- 대칭행렬 eigenvector 직교성.
- Hessian positive semidefinite와 convexity.
- Least squares normal equation.
- PCA eigenvector 유도.
- Bayes theorem 유도.
- SVM margin과 `||w||` 최소화 관계.

### 2순위: 계산형 기본기

작은 숫자가 주어진 문제를 빠르게 풀어야 한다. 계산형은 어려운 이론보다 실수를 줄이는 것이 중요하다.

연습할 계산은 다음과 같다.

- 3개 이하 벡터의 rank 계산.
- 2x2 행렬 eigenvalue/eigenvector 계산.
- 2D vector projection.
- 2x2 covariance matrix 계산.
- 간단한 least squares 해 계산.
- Hessian으로 convex/non-convex 판정.

### 3순위: 알고리즘 설명형

MML 범위에서 알고리즘 설명형으로 나올 가능성이 높은 것은 PCA, EM, SVM이다.

- PCA: centering, covariance/SVD, eigenvector 선택, projection, explained variance.
- EM for GMM: latent variable, responsibility, E-step, M-step, local optimum.
- SVM: hyperplane, margin, support vector, slack variable, dual, kernel.

### 4순위: 비교 서술형

석사 필기시험에서는 여러 개념을 비교하는 문제가 나올 수 있다. 단순 정의보다 “왜 다른지”와 “어디에 쓰이는지”를 써야 한다.

특히 다음 비교는 준비해두면 좋다.

- MLE와 MAP.
- PCA와 linear regression의 projection 차이.
- PCA와 GMM의 latent variable 차이.
- SVD와 eigendecomposition.
- Hard-margin과 soft-margin SVM.
- Parametric density estimation과 discriminative classification.

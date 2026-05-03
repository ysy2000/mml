# Mathematics for Machine Learning Exercises 해설 노트

대상: `mml-book.pdf`의 Part I, Chapter 1-7.

이 파일은 원문 문제를 그대로 옮긴 정답지가 아니라, 시험/복습용으로 빠르게 공부할 수 있도록 핵심 풀이 흐름을 정리한 해설 노트다. 계산 문제는 최종값과 사용한 원리를 함께 적었고, 증명 문제는 어떤 정의를 써야 하는지 중심으로 정리했다.

> 참고: 공개 self-study 자료에는 Chapter 2-7의 end-of-chapter exercise 해설이 주로 정리되어 있다. Chapter 1은 수학 계산 exercise라기보다 책의 동기와 머신러닝 관점을 잡는 장이므로, 아래에는 개념 확인형 답안을 따로 정리했다.

---

## Chapter 1. Introduction and Motivation

### 1.1 머신러닝에서 수학이 필요한 이유

머신러닝은 데이터를 이용해 모델의 성능을 개선하는 방법이다. 여기서 데이터는 벡터와 행렬로 표현되고, 모델 학습은 손실함수 또는 likelihood를 최적화하는 문제로 바뀐다. 따라서 선형대수는 표현 언어, 해석기하는 거리와 projection, 미적분은 gradient 계산, 확률은 불확실성 표현, 최적화는 실제 학습 절차를 담당한다.

핵심은 “알고리즘을 외우는 것”이 아니라, 알고리즘이 어떤 수학 문제를 풀고 있는지 이해하는 것이다.

### 1.2 데이터, 모델, 학습의 의미

- 데이터: 관측된 입력, 출력, feature vector.
- 모델: 데이터의 관계나 생성 과정을 설명하는 수학적 구조.
- 학습: 데이터에 맞게 모델의 파라미터를 조정하는 과정.

예를 들어 선형회귀에서는 데이터가 `X`, `y`로 주어지고, 모델은 `y ≈ X theta`, 학습은 `||y - X theta||^2`를 최소화하는 `theta`를 찾는 것이다.

### 1.3 Part I의 공부 순서

Chapter 2-7은 독립된 과목처럼 보이지만 실제로는 다음 순서로 연결된다.

1. 선형대수: 데이터를 벡터/행렬로 표현한다.
2. 해석기하: 벡터공간에 길이, 각도, 직교성, projection을 부여한다.
3. 행렬분해: 행렬의 숨은 구조를 eigenvalue, SVD로 분해한다.
4. 벡터미적분: 목적함수의 gradient와 Hessian을 계산한다.
5. 확률: 데이터와 모델의 불확실성을 분포로 표현한다.
6. 최적화: 목적함수를 실제로 최소화/최대화한다.

---

## Chapter 2. Linear Algebra

### Exercise 2.1

`R \ {-1}` 위의 연산 `a star b = ab + a + b`가 group인지 확인한다.

핵심은 `(a + 1)(b + 1) = ab + a + b + 1`로 묶는 것이다. `a star b = -1`이 되려면 `(a + 1)(b + 1) = 0`이어야 하므로 `a = -1` 또는 `b = -1`인데, 둘 다 집합에서 제외되어 있다. 따라서 closure가 성립한다.

항등원은 `0`이다. 왜냐하면 `a star 0 = a`이고 `0 star a = a`이기 때문이다. 역원은 `a star y = 0`을 풀면 `y = -a / (a + 1)`이다. 결합법칙은 전개하면 양쪽이 같고, 교환법칙도 실수의 곱셈/덧셈 교환법칙에서 바로 나온다. 따라서 Abelian group이다.

`3 star x star x = 15`는 `x star x = x^2 + 2x`를 넣어 풀면 `4x^2 + 8x - 12 = 0`, 따라서 `x = 1` 또는 `x = -3`.

### Exercise 2.2

`Z_n`에서 덧셈 mod `n`은 Abelian group이다. 잘 정의됨을 보이려면 대표원 `a1 ≡ a2`, `b1 ≡ b2 (mod n)`이면 `(a1 + b1) - (a2 + b2)`도 `n`의 배수임을 보이면 된다.

`Z_5 \ {0}`에서 곱셈 mod 5는 group이다. 항등원은 `1`, 역원은 `1^-1 = 1`, `2^-1 = 3`, `3^-1 = 2`, `4^-1 = 4`.

`Z_8 \ {0}`에서는 closure가 깨진다. 예를 들어 `2 * 4 = 8 ≡ 0 (mod 8)`이고 `0`은 집합에 없다.

일반적으로 `n`이 prime이면 `Z_n \ {0}`는 곱셈 mod `n`에 대해 group이다. 모든 `1, ..., n-1`이 `n`과 서로소이므로 Bezout 정리에 의해 역원이 존재한다. `n`이 composite이면 `n = ab`인 nonzero residue가 있어서 곱이 `0 mod n`이 되므로 group이 아니다.

### Exercise 2.3

상삼각 형태의 `3 x 3` 행렬 집합 `G`는 표준 행렬곱에 대해 group이다.

두 원소를 곱하면 다시 같은 형태가 된다. 항등원은 `I_3`, 역원도 같은 형태로 쓸 수 있다. 결합법칙은 행렬곱의 결합법칙에서 온다. 다만 일반적으로 `AB != BA`이므로 Abelian group은 아니다.

### Exercise 2.4

행렬곱의 정의와 계산 문제다.

- Part a: 앞 행렬의 column 수와 뒤 행렬의 row 수가 맞지 않으면 곱이 정의되지 않는다.
- Part b: 결과는 `[[4, 3, 5], [10, 9, 11], [16, 15, 17]]`.
- Part c: 결과는 `[[5, 7, 9], [11, 13, 15], [8, 10, 12]]`.
- Part d: 결과는 `[[14, 6], [-21, 2]]`.
- Part e: 결과는 `[[12, 3, -3, -12], [-3, 1, 2, 6], [6, 5, 1, 0], [13, 12, 3, 2]]`.

공부 포인트는 “곱셈 가능 조건”과 “결과 행렬의 크기”를 먼저 확인하는 것이다.

### Exercise 2.5

선형방정식계 `Ax = b`를 Gaussian elimination으로 푸는 문제다.

Part a는 row reduction 중 `0 = 1` 꼴의 행이 생긴다. 따라서 `rank(A) < rank([A|b])`이고 해가 없다.

Part b는 free variable이 존재한다. 최종 해는

`x = (alpha + 3, 2alpha, beta, alpha - 1, alpha)^T`, `alpha, beta in R`.

이 유형은 pivot variable과 free variable을 분리해서 해집합을 parametric form으로 쓰면 된다.

### Exercise 2.6

row reduction 결과 다음 식들이 나온다.

- `x2 + x6 = 1`
- `x4 + x6 = -2`
- `x5 - x6 = 1`

따라서 `x1`, `x3`, `x6`를 자유변수로 두면

`x = (alpha, 1 - beta, gamma, -2 - beta, 1 + beta, beta)^T`.

### Exercise 2.7

조건은 eigenvalue `12`에 대한 eigenvector를 찾는 문제와 같다. `A - 12I`의 null space를 구하면 `(3, 3, 2)^T`가 나온다. 성분 합이 1이 되게 normalize하면

`x = (3/8, 3/8, 2/8)^T`.

### Exercise 2.8

Part a: determinant가 `0`이므로 행렬은 invertible하지 않다.

Part b: augmented matrix `[A | I]`를 row reduction하면 오른쪽 블록이 `A^{-1}`이 된다. 핵심 절차는 왼쪽을 `I`로 만드는 것이다.

### Exercise 2.9

Subspace 판정 문제다.

- Part a: `mu^3`는 모든 실수 값을 만들 수 있으므로 집합을 두 parameter의 선형결합으로 쓸 수 있다. basis는 `{(1,1,1)^T, (0,1,-1)^T}`.
- Part b: 제곱항 때문에 scalar multiplication에 닫혀 있지 않다. subspace가 아니다.
- Part c: 원점을 포함하려면 `gamma = 0`이어야 한다. 이때 basis 예시는 `{(3,0,-1)^T, (0,3,2)^T}`.
- Part d: 정수 조건이 있으면 실수배에 닫히지 않는다. subspace가 아니다.

### Exercise 2.10

Part a: 세 벡터를 column으로 둔 행렬의 determinant가 `0`이므로 linearly independent가 아니다.

Part b: `alpha1 x1 + alpha2 x2 + alpha3 x3 = 0`을 성분별로 보면 차례로 `alpha1 = 0`, `alpha2 = 0`, `alpha3 = 0`이 강제된다. 따라서 linearly independent다.

### Exercise 2.11

`y = alpha1 x1 + alpha2 x2 + alpha3 x3`로 두고 연립방정식을 풀면

`alpha1 = -6`, `alpha2 = 3`, `alpha3 = 2`.

따라서 `y = -6x1 + 3x2 + 2x3`.

### Exercise 2.12

두 부분공간 `U1`, `U2`의 교집합 basis를 구한다.

각각의 차원은 `2`, 합공간의 차원은 `3`이므로

`dim(U1 ∩ U2) = dim(U1) + dim(U2) - dim(U1 + U2) = 1`.

계산하면 교집합의 basis로

`{(24, -6, -12, -6)^T}`

를 잡을 수 있다.

### Exercise 2.13

두 행렬의 kernel을 부분공간으로 보는 문제다.

두 행렬 모두 rank가 `2`이고 domain이 `R^3`이므로 nullity는 `1`. 두 kernel의 basis는 모두

`{(1, 1, -1)^T}`.

따라서 `U1 = U2`이고, `U1 ∩ U2`의 basis도 같다.

### Exercise 2.14

이번에는 column space를 부분공간으로 보는 문제다.

각 column space의 차원은 `2`, 합공간 차원은 `3`이므로 교집합 차원은 `1`. 계산하면 basis는

`{(3, 1, 7, 3)^T}`.

### Exercise 2.15

`F`, `G`는 모두 원점을 포함하고 addition/scalar multiplication에 닫혀 있으므로 subspace다.

교집합 조건을 동시에 만족시키면 `a = -3b`가 나오고,

`F ∩ G = span{(2, 1, 3)}`.

다른 방식으로 각 공간의 basis를 잡고 차원공식으로 풀어도 같은 결과가 나온다.

### Exercise 2.16

Linear map 판정 문제다.

- Definite integral: linear. 적분은 덧셈과 scalar multiplication을 보존한다.
- Differentiation: linear.
- 상수항이 붙어 원점을 원점으로 보내지 않는 map: linear가 아니다.
- 행렬곱으로 표현되는 변환: linear.
- 원점 중심 rotation: linear.

Linear인지 빠르게 보려면 항상 `Phi(0) = 0`부터 확인하자.

### Exercise 2.17

선형사상의 행렬표현은

`A_Phi = [[3,2,1], [1,1,1], [1,-3,0], [2,3,1]]`.

rank가 `3`이므로 kernel은 `{0}`이고 image의 차원은 `3`이다. 즉 `dim ker(Phi) = 0`, `dim Im(Phi) = 3`.

### Exercise 2.18

Automorphism은 선형이고 bijective인 자기 사상이다. 따라서 kernel은 `{0}`, image는 전체공간 `E`다.

그러므로 `ker(f) ∩ Im(g) = {0}`이고, 합성 `g ∘ f`도 bijective이므로 kernel은 `{0}`, image는 `E`.

### Exercise 2.19

Part a: 주어진 행렬의 rank가 `3`이면 `ker(Phi) = {0}`, `Im(Phi) = R^3`.

Part b: basis change matrix `P`를 만들고

`A_bar = P^{-1} A_Phi P`

를 계산한다. 결과는

`[[6, 9, 1], [-3, -5, 0], [-1, -1, 0]]`.

### Exercise 2.20

Basis change와 선형사상 행렬표현 문제다.

- `B`, `B'`는 모두 `R^2`의 basis다.
- `B'`에서 `B`로 가는 change-of-basis matrix는 `P1 = [[4, 0], [6, -1]]`.
- `C`는 determinant가 `4`라서 `R^3`의 basis다.
- `P2 = [[1,0,1], [2,-1,0], [-1,2,-1]]`.
- 선형사상의 행렬은 `A_Phi = [[1,-1], [0,1], [2,-1]]`.
- 새 basis에서의 행렬은 `A' = P2 A_Phi P1 = [[0,2], [-10,3], [12,-4]]`.
- 예시 벡터 `(2,3)^T`에 적용하면 결과는 `(6,-11,12)^T`.

---

## Chapter 3. Analytic Geometry

### Exercise 3.1

주어진 bilinear form이 inner product인지 확인한다.

확인할 조건은 세 가지다.

1. Bilinear: 각 argument에 대해 선형.
2. Symmetric: `<x,y> = <y,x>`.
3. Positive definite: `x != 0`이면 `<x,x> > 0`.

계산하면 `<x,x> = x1^2 - 2x1x2 + 2x2^2 = (x1 - x2)^2 + x2^2`이다. 이는 `x != 0`이면 양수이므로 inner product다.

### Exercise 3.2

주어진 form은 symmetric하지 않다. 예를 들어 `<e1,e2> = 0`인데 `<e2,e1> = 1`이면 inner product가 아니다.

행렬로 표현했을 때 inner product가 되려면 matrix가 symmetric positive definite여야 한다.

### Exercise 3.3

거리는 `d(x,y) = sqrt(<x-y, x-y>)`로 계산한다.

- Part a: `sqrt(22)`.
- Part b: `sqrt(47)`.

### Exercise 3.4

각도는

`omega = arccos(<x,y> / (||x|| ||y||))`.

- Part a: `arccos(-3 / sqrt(10))`.
- Part b: `arccos(-11 / sqrt(126))`.

### Exercise 3.5

Projection은 basis matrix `B`를 잡고 normal equation을 푼다.

`B^T B lambda = B^T x`

를 풀면 `lambda = (-3, 4, 1)^T`, 따라서

`pi_U(x) = (1, -5, -1, -2, 3)^T`.

거리는 residual norm이다.

`d(x,U) = ||x - pi_U(x)|| = sqrt(60) = 2sqrt(15)`.

### Exercise 3.6

Projection `pi_U(e2)`를 `lambda1 e1 + lambda2 e3` 꼴로 두고 residual이 `U`에 직교한다는 조건을 쓴다.

계산하면 `lambda1 = 1/2`, `lambda2 = -1/2`이고,

`pi_U(e2) = (1/2, 0, -1/2)^T`.

거리 `d(e2,U) = 1`.

### Exercise 3.7

`pi`가 projection이면 `pi^2 = pi`.

Part a: `(id - pi)^2 = id - 2pi + pi^2`. 이것이 `id - pi`와 같으려면 `pi^2 = pi`여야 한다. 따라서 `id - pi`도 projection이다.

Part b:

- `Im(id - pi) = ker(pi)`.
- `ker(id - pi) = Im(pi)`.

증명은 각각 원소를 하나 잡아 양쪽 포함관계를 보이면 된다.

### Exercise 3.8

Gram-Schmidt로 orthonormal basis를 만든다.

첫 벡터는 `b1 = (1,1,1)^T`. 두 번째는

`b2' = b2 - proj_span(b1)(b2) = (-4/3, 5/3, -1/3)^T`.

정규화하면 orthonormal basis는

`{(1/sqrt(3))(1,1,1)^T, (1/sqrt(42))(-4,5,-1)^T}`.

### Exercise 3.9

Cauchy-Schwarz inequality를 쓰는 문제다.

Part a: `x = (x1,...,xn)`, `y = (1,...,1)`에 대해

`sum xi <= sqrt(sum xi^2) sqrt(n)`.

`sum xi = 1`을 넣고 제곱하면 원하는 부등식이 나온다.

Part b: `x = (sqrt(x1),...,sqrt(xn))`, `y = (1/sqrt(x1),...,1/sqrt(xn))`를 두고 Cauchy-Schwarz를 적용한다.

### Exercise 3.10

2D rotation matrix는

`R = [[cos theta, -sin theta], [sin theta, cos theta]]`.

`theta = 30°`이면

`R = (1/2)[[sqrt(3), -1], [1, sqrt(3)]]`.

따라서 주어진 벡터에 `R`을 곱하면 된다. 회전 방향이 명시되지 않으면 보통 반시계방향을 기준으로 잡는다.

---

## Chapter 4. Matrix Decompositions

### Exercise 4.1

Determinant 계산 문제다.

Laplace expansion 또는 Sarrus rule을 쓰면 모두 `det(A) = 0`이 나온다.

### Exercise 4.2

Gaussian elimination으로 determinant를 계산한다. row replacement는 determinant를 바꾸지 않는다. 삼각행렬로 만든 뒤 대각성분을 곱하면 된다.

결과는 `det(A) = 6`.

### Exercise 4.3

Eigenvalue/eigenvector 계산 문제다.

- Part a: eigenvalue는 `lambda = 1`, eigenspace는 `span{(0,1)}`.
- Part b: eigenvalue는 `2`, `-3`. eigenspace는 `E_2 = span{(1,2)}`, `E_-3 = span{(-2,1)}`.

### Exercise 4.4

Characteristic polynomial은 `(lambda - 2)(lambda - 1)(lambda + 1)^2`.

Eigenspace는

- `E_2 = span{(1,0,1,1)}`
- `E_1 = span{(1,1,1,1)}`
- `E_-1 = span{(0,1,1,0)}`

`lambda = -1`은 algebraic multiplicity가 2지만 eigenspace dimension은 1이다.

### Exercise 4.5

Diagonalizable 여부와 invertible 여부는 별개다.

첫 두 행렬은 diagonal matrix라서 diagonalizable이다. 나머지 두 행렬은 충분한 수의 linearly independent eigenvector가 없어서 diagonalizable이 아니다.

Invertible 여부는 determinant로 본다. determinant가 nonzero인 행렬만 invertible이다.

### Exercise 4.6

Part a: characteristic polynomial은 `(5 - lambda)(1 - lambda)^2`. `lambda = 1`에 대한 linearly independent eigenvector가 하나뿐이므로 diagonalizable하지 않다.

`E_5 = span{(1,1,0)}`, `E_1 = span{(-3,1,0)}`.

Part b: eigenvalue는 `1`과 `0`. `lambda = 0`에 대해 3개의 linearly independent eigenvector가 있어 diagonalizable하다.

`E_1 = span{(1,0,0,0)}`, `E_0 = span{(1,-1,0,0), (0,0,1,0), (0,0,0,1)}`.

### Exercise 4.7

- Part a: 실수 eigenvalue가 없으므로 `R` 위에서는 diagonalizable하지 않다. `C` 위에서는 가능하다.
- Part b: symmetric matrix이므로 diagonalizable. eigenvalue는 `3`, `0`, `0`.
- Part c: eigenvalue multiplicity와 eigenspace dimension이 맞지 않아 diagonalizable하지 않다.
- Part d: 충분한 eigenvector가 있으므로 diagonalizable. diagonal form은 `diag(1,2,2)`.

### Exercise 4.8

SVD 계산 문제다.

먼저 `A^T A`를 구하고 eigendecomposition한다. 고윳값은 `25`, `9`, `0`이므로 singular value는 `5`, `3`.

`Sigma = [[5,0,0], [0,3,0]]`.

`V`는 `A^T A`의 orthonormal eigenvectors, `U`는 `u_i = Av_i / sigma_i`로 구한다.

### Exercise 4.9

Eigenvalue가 complex라 eigendecomposition은 실수에서 바로 쓰기 어렵지만, SVD는 항상 가능하다.

`A^T A = [[5,3], [3,5]]`의 eigenvalue는 `8`, `2`. 따라서 singular value는 `2sqrt(2)`, `sqrt(2)`.

이 문제의 경우 `U = I`가 된다.

### Exercise 4.10

Rank-one decomposition은 SVD의 각 항

`A_i = sigma_i u_i v_i^T`

를 쓰면 된다.

Exercise 4.8의 SVD를 이용하면 `A = A1 + A2`이고, 각각 rank-one matrix다.

### Exercise 4.11

`A^T A`와 `A A^T`의 nonzero eigenvalues가 같음을 보이는 문제다.

`A^T A x = lambda x`, `lambda != 0`이면

`A A^T (A x) = A(A^T A x) = lambda A x`.

또한 `lambda != 0`이면 `Ax != 0`이다. 따라서 `Ax`는 `A A^T`의 eigenvector이고 eigenvalue는 `lambda`다.

### Exercise 4.12

`max ||Ax||_2 / ||x||_2`는 행렬 `A`가 어떤 방향을 가장 크게 늘리는지를 묻는 값이다.

SVD `A = U Sigma V^T`에서 `U`, `V^T`는 norm을 보존하는 회전/반사이고, scaling은 `Sigma`만 담당한다. 따라서 최대 scaling은 가장 큰 singular value다.

---

## Chapter 5. Vector Calculus

### Exercise 5.1

`f(x) = 4 log(x) sin(x^3)`.

Product rule과 chain rule을 쓰면

`f'(x) = (4/x) sin(x^3) + 12x^2 log(x) cos(x^3)`.

### Exercise 5.2

Sigmoid `f(x) = (1 + exp(-x))^-1`.

미분하면

`f'(x) = exp(-x) / (1 + exp(-x))^2`.

또는 `f'(x) = f(x)(1 - f(x))`.

### Exercise 5.3

Gaussian density를 `x`에 대해 미분하면

`f'(x) = ((mu - x) / sigma^2) f(x)`.

핵심은 exponential 내부를 chain rule로 미분하는 것이다.

### Exercise 5.4

0에서의 5차 Taylor polynomial:

`T5(x) = 1 + x - (1/2)x^2 - (1/6)x^3 + (1/24)x^4 + (1/120)x^5`.

낮은 차수의 Taylor polynomial은 여기서 뒷항을 자르면 된다.

### Exercise 5.5

Jacobian의 차원과 계산 문제다.

- `df1/dx`의 차원은 `1 x 2`.
- `df2/dx`의 차원은 `1 x n`.
- `df3/dx`는 `xx^T`를 vectorize하면 `n^2 x n` Jacobian이 된다.

계산:

- `df1/dx = [cos(x1)cos(x2), -sin(x1)sin(x2)]`.
- `df2/dx = y^T`.
- `d(xx^T)/dx`는 3차 tensor로 보는 것이 자연스럽고, vectorize하면 각 entry `x_i x_j`를 각 `x_k`로 미분한 행렬이 된다.

### Exercise 5.6

`f(t) = sin(log(t^T t))`이면

`df/dt = cos(log(t^T t)) * 2t^T / (t^T t)`.

`g(X) = tr(AXB)`이면 entry-wise 계산 또는 trace trick으로

`dg/dX = A^T B^T`.

### Exercise 5.7

Chain rule 문제다.

Part a:

`f(x) = log(1 + x^T x)`이면

`df/dx = 2x^T / (1 + x^T x)`.

Part b:

`z = Ax + b`, `f(z) = sin(z)`이면

`df/dx = diag(cos(z)) A`.

### Exercise 5.8

Part a:

`f = exp(-1/2 y^T S^{-1} y)`, `y = x - u`.

`df/dx = -1/2 exp(-1/2 y^T S^{-1}y) * y^T(S^{-1} + S^{-T})`.

`S`가 symmetric이면 더 단순하게 `-exp(...) y^T S^{-1}`.

Part b:

`tr(xx^T + sigma^2 I) = x^T x + n sigma^2`이므로

`df/dx = 2x^T`.

Part c:

`f(z) = tanh(z)`, `z = Ax + b`.

`df/dx = diag(1 / cosh^2(z_i)) A`.

### Exercise 5.9

Reparameterization 형태의 chain rule이다.

`g(nu) = log p(x, t(epsilon,nu)) - log q(t(epsilon,nu), nu)`.

따라서

`dg/dnu = [p'(x,t) t'(epsilon,nu)] / p(x,t) - [q'(t,nu) t'(epsilon,nu)] / q(t,nu)`.

정확한 형태는 `q`가 `nu`에 직접 의존하는 항도 포함해야 하므로, 실제 계산에서는 total derivative를 써야 한다.

---

## Chapter 6. Probability and Distributions

### Exercise 6.1

Joint probability table에서 marginal과 conditional을 계산한다.

Marginal:

- `p(x) = (0.16, 0.17, 0.11, 0.22, 0.34)`.
- `p(y) = (0.26, 0.47, 0.27)`.

Conditional:

- `p(x | y1) ≈ (0.038, 0.077, 0.115, 0.385, 0.385)`.
- `p(y | x3) ≈ (0.273, 0.454, 0.273)`.

공식은 `p(x|y) = p(x,y) / p(y)`.

### Exercise 6.2

Gaussian mixture의 marginal, mean, mode, median 문제다.

Mixture marginal은 각 component marginal의 weighted sum이다.

`p(x) = 0.4 N(x | 10, 1) + 0.6 N(x | 0, 8.4)`.

`p(y) = 0.4 N(y | 2, 1) + 0.6 N(y | 0, 1.7)`.

Mean은 weight를 곱해 더하면 된다.

`E[x] = 4`, `E[y] = 0.8`, 따라서 joint mean은 `(4, 0.8)^T`.

Mixture의 mode와 median은 일반적으로 closed form이 없으므로 derivative 조건 또는 CDF 조건을 수치적으로 풀어야 한다.

### Exercise 6.3

Bernoulli likelihood의 conjugate prior는 Beta distribution이다.

Prior:

`p(mu | alpha, beta) ∝ mu^(alpha-1)(1-mu)^(beta-1)`.

Data likelihood:

`p(x1,...,xN | mu) = mu^(sum xi)(1-mu)^(N - sum xi)`.

Posterior:

`p(mu | data) ∝ mu^(alpha + sum xi - 1)(1-mu)^(beta + N - sum xi - 1)`.

즉 posterior는

`Beta(alpha + sum xi, beta + N - sum xi)`.

### Exercise 6.4

Bayes theorem 문제다.

Bag 1: `p(mango|1)=2/3`, `p(apple|1)=1/3`.

Bag 2: `p(mango|2)=1/2`, `p(apple|2)=1/2`.

Prior: `p(1)=0.6`, `p(2)=0.4`.

망고를 뽑았을 때 Bag 2였을 확률:

`p(2|mango) = [(1/2)(0.4)] / [(2/3)(0.6) + (1/2)(0.4)] = 1/3`.

### Exercise 6.5

Linear Gaussian state-space model 문제다.

Part a: Gaussian은 선형변환과 Gaussian noise의 합에 닫혀 있으므로 `p(x0,...,xT)`는 Gaussian이다.

Prediction:

`x_{t+1} | y_{1:t} ~ N(A mu_t, A Sigma_t A^T + Q)`.

Observation:

`y_{t+1} | x_{t+1} ~ N(C x_{t+1}, R)`.

Predictive observation:

`y_{t+1} | y_{1:t} ~ N(C A mu_t, C(A Sigma_t A^T + Q)C^T + R)`.

Posterior는 Bayes theorem으로

`p(x_{t+1}|y_{1:t+1}) ∝ p(y_{t+1}|x_{t+1}) p(x_{t+1}|y_{1:t})`.

### Exercise 6.6

Variance identity:

`Var[x] = E[(x - mu)^2] = E[x^2] - (E[x])^2`.

전개하면 `E[x^2 - 2xmu + mu^2] = E[x^2] - 2muE[x] + mu^2 = E[x^2] - mu^2`.

### Exercise 6.7

Pairwise squared difference identity:

`(1/N^2) sum_{i,j} (xi - xj)^2`

를 전개하면

`(2/N) sum_i xi^2 - 2((1/N)sum_i xi)^2`.

즉 모든 pairwise distance 평균은 variance와 직접 연결된다.

### Exercise 6.8

Bernoulli distribution은 exponential family로 쓸 수 있다.

`p(x|mu) = mu^x(1-mu)^(1-x)`

`= exp{x log(mu/(1-mu)) + log(1-mu)}`.

Natural parameter:

`theta = log(mu/(1-mu))`.

Log-partition:

`A(theta) = log(1 + exp(theta))`.

### Exercise 6.9

Binomial도 exponential family다.

`p(x|N,mu) = C(N,x) mu^x(1-mu)^(N-x)`.

Natural parameter는 Bernoulli와 같이

`theta = log(mu/(1-mu))`,

`A(theta) = N log(1 + exp(theta))`.

Beta distribution도 exponential family로 표현 가능하다.

`theta1 = alpha - 1`, `theta2 = beta - 1`,

sufficient statistics는 `log x`, `log(1-x)`.

### Exercise 6.10

두 Gaussian의 곱은 normalization constant를 제외하면 다시 Gaussian이다.

`N(x|a,A) N(x|b,B) = c N(x|c,C)`.

여기서

- `C^{-1} = A^{-1} + B^{-1}`.
- `c_vec = C(A^{-1}a + B^{-1}b)`.
- normalization constant는 `N(a|b,A+B)` 또는 `N(b|a,A+B)` 형태로 쓸 수 있다.

핵심 기법은 exponent의 quadratic form을 completing the square로 정리하는 것이다.

### Exercise 6.11

Law of total expectation:

`E_Y[E_X[x|y]] = E_X[x]`.

증명:

`E_Y[E_X[x|y]] = ∫∫ x p(x|y)p(y) dx dy = ∫∫ x p(x,y) dx dy = ∫ x p(x) dx`.

### Exercise 6.12

Linear Gaussian transformation 문제다.

`y = Ax + b + w`, `w ~ N(0,Q)`이면

`p(y|x) = N(y | Ax + b, Q)`.

만약 `x ~ N(mu_x, Sigma_x)`이면

`y ~ N(A mu_x + b, A Sigma_x A^T + Q)`.

또 `z = Cy + noise`이면

`z ~ N(C(A mu_x + b), C(A Sigma_x A^T + Q)C^T + R)`.

Posterior는

`p(x|y) = p(y|x)p(x) / p(y)`.

### Exercise 6.13

Probability integral transform:

`y = F_X(x)`이면 `y`는 `[0,1]`에서 uniform distribution을 따른다.

변수변환 공식으로

`f_Y(y) = f_X(x) |dx/dy| = f_X(x) / |dF_X(x)/dx| = f_X(x)/f_X(x) = 1`.

---

## Chapter 7. Continuous Optimization

### Exercise 7.1

Stationary point와 second derivative test 문제다.

`f'(x) = 3x^2 + 12x - 3`, `f''(x) = 6x`.

`f'(x) = 0`을 풀면

`x = -2 ± sqrt(5)`.

`f''(-2 + sqrt(5)) > 0`이므로 local minimum, `f''(-2 - sqrt(5)) < 0`이므로 local maximum.

### Exercise 7.2

Full-batch gradient descent:

`theta_{i+1} = theta_i - gamma_i sum_{n=1}^N grad L_n(theta_i)^T`.

Mini-batch size 1, 즉 stochastic gradient descent:

`theta_{i+1} = theta_i - gamma_i grad L_n(theta_i)^T`,

여기서 `n`은 매 step에서 무작위로 뽑은 data index다.

### Exercise 7.3

Convex set 관련 참/거짓.

- 두 convex set의 intersection은 convex다. 두 점을 잇는 선분이 두 집합 모두에 있으므로 교집합에도 있다.
- Union은 일반적으로 convex가 아니다. 서로 떨어진 두 convex set의 합집합을 생각하면 된다.
- Set difference도 일반적으로 convex가 아니다. 중간 선분 일부가 제거될 수 있다.

### Exercise 7.4

Convex function 관련 참/거짓.

- `f + g`는 convex다. convex inequality를 더하면 된다.
- `f - g`는 일반적으로 convex가 아니다.
- `fg`도 일반적으로 convex가 아니다.
- Maximum point 자체는 함수가 아니므로 convex function이라고 말할 수 없다. 다만 argmax set이 singleton이면 집합으로서는 convex다.

### Exercise 7.5

Linear programming standard form으로 바꾸는 문제다.

`y = (x0, x1, xi)^T`, `c = (p0, p1, 1)^T`로 두면 objective는 `c^T y`.

Constraint는 `Ay <= b`로 정리할 수 있고,

`max_y c^T y subject to Ay <= b`

꼴의 standard linear program이 된다.

### Exercise 7.6

Linear program의 dual을 구한다.

Primal을

`min_x c^T x subject to Ax <= b`

로 쓰고 Lagrangian

`L(x,lambda) = c^T x + lambda^T(Ax - b)`

를 둔다.

`x`에 대해 infimum이 finite하려면

`c + A^T lambda = 0`, `lambda >= 0`.

Dual은

`max_lambda -b^T lambda subject to A^T lambda + c = 0, lambda >= 0`.

### Exercise 7.7

Quadratic program의 dual.

Primal:

`min_x 1/2 x^T Q x + c^T x subject to Ax <= b`.

Lagrangian:

`L(x,lambda) = 1/2 x^T Q x + c^T x + lambda^T(Ax - b)`.

Stationarity:

`Qx + c + A^T lambda = 0`,

따라서

`x = -Q^{-1}(c + A^T lambda)`.

Dual은 이 값을 다시 Lagrangian에 대입해 얻는다.

### Exercise 7.8

Primal:

`min_w 1/2 w^T w subject to 1 - x^T w <= 0`.

Lagrangian:

`L(w,lambda) = 1/2 w^T w + lambda(1 - x^T w)`.

Stationarity:

`w = lambda x`.

Dual function:

`D(lambda) = -1/2 lambda^2 x^T x + lambda`, with `lambda >= 0`.

### Exercise 7.9

Convex conjugate:

`f*(s) = sup_x s^T x - f(x)`.

미분해서 최적점을 찾으면 `s_d = log x_d - 1`, 따라서

`x_d = exp(s_d + 1)`.

이를 대입하면

`f*(s) = - sum_d exp(s_d + 1)`.

주의: 사용하는 `f(x)` 정의에 따라 부호가 달라질 수 있으므로 원문 함수의 domain과 convexity를 확인해야 한다.

### Exercise 7.10

Quadratic function의 convex conjugate.

`f(x) = 1/2 x^T A x + b^T x + c`, `A` positive definite라고 하면

Stationarity:

`s = Ax + b`, 따라서 `x = A^{-1}(s - b)`.

대입하면

`f*(s) = 1/2 (s - b)^T A^{-1}(s - b) - c`.

### Exercise 7.11

Hinge loss `L(alpha) = max(0, 1 - alpha)`의 convex conjugate.

정의에 따라 구간을 나누면

`L*(beta) = beta` if `-1 <= beta <= 0`,

`L*(beta) = +infty` otherwise.

Regularized conjugate 형태는 `beta in [-1,0]`에서 quadratic maximization을 풀면 된다. 핵심은 domain restriction이 hinge loss의 dual constraint로 이어진다는 점이다.

---

## 빠르게 복습하는 순서

1. Chapter 2에서 `rank`, `basis`, `kernel`, `image`, `basis change`를 먼저 잡는다.
2. Chapter 3에서 `inner product`, `projection`, `orthogonality`를 잡는다.
3. Chapter 4에서 `eigenvalue`, `diagonalization`, `SVD`의 계산 절차를 익힌다.
4. Chapter 5에서 chain rule과 matrix derivative의 모양을 외운다.
5. Chapter 6에서 Bayes theorem, Gaussian, expectation identity를 반복한다.
6. Chapter 7에서 convexity, Lagrangian, dual의 표준 흐름을 익힌다.

시험 직전에는 답을 외우기보다 “이 문제는 어떤 정의를 쓰는가?”를 먼저 말할 수 있어야 한다.

import numpy as np

def calc_P_fix_sigma(X, sigma_sq):
    assert X.ndim == 2
    n = len(X)

    # 距離
    D = np.zeros((n,n))
    # 条件付き確率
    Pcond = np.zeros(D.shape)
    # シャノンエントロピー
    H = np.zeros(n)

    for i in range(n):
        # 距離の計算
        for j in range(n):
            # 本来はsigma_sqは探索で求めるが、既知のものとする
            D[i,j]=np.exp(-np.sum((X[i,:]-X[j,:])**2, axis=-1) / (2*sigma_sq))

        # 条件付き確率の計算
        for j in range(n):
            if i==j: continue
            tmp = 0
            for k in range(n):
                if i==k: continue
                tmp += D[i,k]
            # p j|i
            Pcond[i,j] = D[i,j] / tmp

        # シャノンエントロピーの計算
        tmp = 0
        for j in range(n):
            if i==j: continue
            tmp += Pcond[i,j] * np.log2(Pcond[i,j])
        H[i] = -tmp

    # perplexity＝クラスタの大きさ
    perp = np.power(2.0, H)
    # perplexityとsigma_sqは正の関係がある
    # perplexityはsigma_sqを大きくすると最終的にn-1になる
    # 実際はperp[i]を固定して、それを満たすようなsigma_sqを探す

    # 同時確率
    Pjoint = np.zeros(D.shape)
    for i in range(n):
        for j in range(n):
            Pjoint[i,j] = (Pcond[i,j]+Pcond[j,i]) / 2 / n

    print("条件付き確率")
    print(Pcond)
    print("同時確率")
    print(Pjoint)
    print("perplexity")
    print(perp)

# ベクトル版
def X_to_affinities(X):
    # 距離のsigma_sqで割らない部分を先に計算
    gramX = np.dot(X, X.T) # グラム行列をとり(N,N)にする
    # ||xi - xj||^2 = ||xi||^2 - 2xixj + ||xj||^2
    Xi_sq = np.sum(X**2, axis=-1).reshape(-1, 1)
    xj_sq = np.sum(X**2, axis=-1).reshape(1, -1)
    affinities = Xi_sq + xj_sq - 2*gramX
    return affinities

def affinities_to_perp(affnities, i, sigma_sq):
    assert affnities.ndim == 2
    n = affnities.shape[0]
    assert i >= 0 and i < n
    eps = 1e-14

    # 距離の計算
    # D=np.exp(-np.sum((X[i:i+1,:]-X)**2, axis=-1) / (2*sigma_sq)) だから
    # np.sum(…)をaffinitiesとすれば、
    # D = np.power(np.exp(-affinities[i]), 1/(2*sigma_sq)) すれば高速に計算できる（アフィンのキャッシュ）
    D = np.power(np.exp(-affnities[i]), 1/(2*sigma_sq))

    # 条件付き確率の計算
    # i≠kの和
    flag = np.ones(n)
    flag[i] = 0
    d_sum = np.sum(D*flag)
    # Pcondの対角成分も0にする
    Pcond = D / (d_sum+eps) # 0割り対策
    Pcond[i] = 0

    # シャノンエントロピーの計算
    H = -np.sum(Pcond * np.log2(Pcond+eps)) # log0対策
    # perplexity
    perp = np.power(2.0, H)
    return Pcond, perp

def calc_P_fix_sigma_vectorized(X, sigma_sq):
    assert X.ndim == 2
    n = len(X)

    Pcond = np.zeros((n,n))
    perp = np.zeros(n)
    affinites = X_to_affinities(X)

    for i in range(n):
        Pcond[i,:], perp[i] = affinities_to_perp(affinites, i, sigma_sq)

    print(Pcond)
    print(perp)

    return Pcond

from scipy.stats import entropy

def eval_entropies(P, Q):
    print("KLダイバージェンス")
    print(entropy(P, Q))

# Perplexity
def eval_perplexity(X):
    print("sigma=0.5の場合")
    calc_P_fix_sigma_vectorized(X,0.5)
    print("sigma=1の場合")
    calc_P_fix_sigma_vectorized(X,1)
    print("sigma=2の場合")
    calc_P_fix_sigma_vectorized(X,2)
    print("sigma=10の場合")
    calc_P_fix_sigma_vectorized(X,10)

from sklearn.manifold import _utils

# sigma^2の探索
def find_sigma(X, target_perplexity):
    # 条件付き確率を求める
    affinites = X_to_affinities(X).astype(np.float32)
    Pcond = _utils._binary_search_perplexity(affinites, None, target_perplexity, 1)
    # Perplexityの確認
    eps = 1e-14
    H = -np.sum(Pcond * np.log2(Pcond+eps), axis=-1)
    perp = np.power(2.0, H)
    print(Pcond)
    print(perp)

if __name__ == "__main__":
    X = np.array([1,2,3,4]).reshape(-1,1)
    eval_perplexity(X)
    #find_sigma(X, 2)

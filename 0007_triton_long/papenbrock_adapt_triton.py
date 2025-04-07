import pennylane as qml
from jax import numpy as np
import jax
import scipy
# import numpy as np
# jax.config.update("jax_platform_name", "cpu")
# jax.config.update('jax_enable_x64', True)

import random
import optax
import tqdm
import matplotlib.pyplot as plt
import os





def main():

    # N = 16 # deuteron
    N = 24 # Triton

    if N==16:
        from papenbrock_ham_deuteron import get_info
        # H = qml.utils.sparse_hamiltonian(Hamiltonian)
        # H = Hamiltonian.sparse_matrix()
        #
        # gs = scipy.sparse.linalg.eigs(H,k=1,which='SR',return_eigenvectors=False)
        gs = -2.483
    else:
        from papenbrock_ham_triton import get_info
        gs = -72

    print('starting')
    Hamiltonian, pools = get_info()

    # H = qml.utils.sparse_hamiltonian(Hamiltonian)
    # H = Hamiltonian.sparse_matrix()
    # gs = scipy.sparse.linalg.eigs(H,k=1,which='SR',return_eigenvectors=False) # = -72.
    # print(gs)
    print('size pool ',len(pools))

    dev = qml.device("lightning.gpu", wires=range(N))
    # dev = qml.device("lightning.qubit", wires=range(N))# on cpu

    @qml.qnode(dev, interface="jax")
    def vqe(tau,ops,H):

        qml.X(0)
        qml.X(1)
        qml.X(2)
        for _,op in enumerate(ops):
            qml.TrotterProduct(op,time=tau[_],n=1)
        return qml.expval(H)



    key = jax.random.key(42)
    opt = optax.adam(learning_rate=0.1)
    theta = []

    max_iter = 300
    operators = []
    energy = []
    indices = []
    # print(pools)
    ### VQE ###
    # operators = pools
    # theta = jax.random.normal(key,len(pools))

    print('starting')
    for i in range(max_iter):
        ### ADAPT ###
        ## comment all of this if you wish to do full vqe
        gradients = []
        # random.shuffle(operators)
        print(vqe(theta,operators,Hamiltonian))
        for op in pools:
            theta_ = list(theta) + list(0*jax.random.normal(key,1)/10000)

            theta_ = jax.numpy.array(theta_)
            operators_ = operators.copy() + list([op])

            gradient = jax.grad(lambda x: vqe(x,operators_,Hamiltonian)[0][0])(theta_)
            # print(gradient)

            if len(gradient)>1:
                gradient = [gradient[-1]]

            gradients.append(gradient)

        gradients = np.absolute(np.array(gradients).reshape(-1))
        nbr = 90

        sorted_index = np.argsort(gradients)[-nbr:]
        # print((gradients))
        # sorted_index = [np.argmax(gradients)]
        print(f"Adding all {nbr} operators -> VQE")
        for  index in sorted_index:
            indices.append(index)
            op = pools[index]
            # print('adding operator {} with gradient {} '.format(index, gradients[index]))
            operators.append(op)
            key = jax.random.key(42+i)
            theta =  list(theta.copy()) + list(0*jax.random.normal(key,1)/10000)

        theta = jax.numpy.array(theta)

        ### ADAPT ####


        opt = optax.adam(learning_rate=0.01)
        opt_state = opt.init(theta)

        pbar = tqdm.tqdm(range(100), desc="Optimizing")
        for ii in range(100):
            gradient = jax.grad(lambda x: vqe(x,operators,Hamiltonian)[0][0])(theta)

            updates, opt_state = opt.update(gradient, opt_state)
            theta = optax.apply_updates(theta, updates)

            energy.append(vqe(theta,operators,Hamiltonian)[0][0])
            pbar.set_description(f"Best cost: {np.min(np.array(energy)):.6f}, {energy[-1]}")
            if i>10:
                if abs(energy[-1]-energy[-5])<10**-4:
                    break


        print(energy[-1],np.min(np.array(energy)))
        plt.figure()
        plt.plot(energy)
        plt.savefig('lattice_energy_optimization_triton.png')
        np.save('results/nuclear/energy_triton.npy',energy)
        np.save('results/nuclear/indices_triton.npy',indices)


if __name__ == '__main__':
    main()

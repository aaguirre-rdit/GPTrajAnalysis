import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import seaborn as sn
from sklearn.cluster import DBSCAN
from mpl_toolkits.mplot3d import Axes3D
from sklearn import metrics
import pickle
import pandas as pd

class GPTrajAnalysis(object):
    def __init__(self,traj_fn,top_fn,chunk,stride=1):
        self.top=top
        try:
            self.traj=md.iterload(traj_fn,top=top_fn,chunk=chunk,stride=stride)
            return "Trajectory was loaded succesfully."
        except (ValueError,IndexError,SyntaxError):
            return "Load a trajectory first."
    def calculate_and_filter_by_dist(self,atom_A_name,atom_B_name,distance_cutoff):
        self.coordinates_array=[]
        coords={
            "frame":[],
            "atom_A_index":[],
            "X":[],
            "Y":[],
            "Z":[]
        }
        for chunk_ind,chunk in enumerate(self.traj):
            top, bonds = chunk.top.to_dataframe()
            atom_A_top=top[top.name==atom_A_name]
            atom_B_top=top[top.name==atom_B_name]
            for f_ind, frame in enumerate(chunk):
                for atom_A_ind in atom_A_top.index:
                    # print vco_ind
                    atom_pairs = np.array([[atom_A_ind, j] for j in atom_B_top.index])
                    # print atom_pairs
                    distances = md.compute_distances(frame, atom_pairs)
                    xyz = frame.xyz[:, atom_A_ind, :]

                    if len(distances[distances <= distance_cutoff]):
                        coords["frame"].extend((chunk * chunk_index) + f_ind)
                        coords["atom_A_index"].extend(atom_A_ind)
                        coords["X"].extend(xyz[0,0])
                        coords["Y"].extend(xyz[0, 1])
                        coords["Z"].extend(xyz[0, 2])
                        self.coordinates_array.append([(chunk * chunk_index) + f_ind, \
                                                       atom_A_ind, xyz[0, 0], xyz[0, 1], xyz[0, 2]])

        self.coordenates=pd.DataFrame(coords)
        pickle.dump(coords,open("coordenates.p","wb"),protocol=2)
        return self.coordenates

    def DBSCAN(self,eps,minsamples):
        xyz_nots=np.array(self.coordinates_array[:,2:])
        X = xyz_nots
        db = DBSCAN(eps=eps, min_samples=minsamples).fit(X)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        self.labels = db.labels_
        self.unique_labels = set(self.labels)
        self.n_clusters_ = len(set(self.labels)) - (1 if -1 in self.labels else 0)
        self.cluster_xyz = {}
        self.coordinates_n_labels=np.column_stack((self.coordinates_array, self.labels))

        for ulbl in self.unique_labels:
            cluster_xyz[ulbl] = []
            for ind, (coord, lbl) in enumerate(zip(self.coordinates_array, self.labels)):
                if lbl == ulbl:
                    self.cluster_xyz[ulbl].append([coord[2], coord[3], coord[4]])
        return (self.coordinates_n_labels,self.cluster_xyz)

        def plot2d(self):
            fig, ax = plt.subplots(1, 2, figsize=(14, 7))
            # Black removed and is used for noise instead.

            colors = [plt.cm.hsv(each)
                  for each in np.linspace(0, 1, len(self.unique_labels))]
            for k, col in zip(self.unique_labels, colors):
                class_member_mask = (self.labels == k)
                if k != -1:
                    xy = X[class_member_mask & core_samples_mask]
                    ax[0].plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                            markeredgecolor='k', markersize=14)
                    ax[1].plot(xy[:, 2], xy[:, 1], 'o', markerfacecolor=tuple(col),
                            markeredgecolor='k', markersize=14)
                    xy = X[class_member_mask & ~core_samples_mask]
                    ax[0].plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                            markeredgecolor='k', markersize=6)
                    ax[1].plot(xy[:, 2], xy[:, 1], 'o', markerfacecolor=tuple(col),
                            markeredgecolor='k', markersize=6)
            ax[0].set_xlabel("x")
            ax[1].set_xlabel("z")
            ax[0].set_ylabel("y")
            plt.suptitle('Estimated number of clusters: %d' % self.n_clusters_)
            plt.subplots_adjust(wspace=0, hspace=0)
            ax[1].set_yticks([])
            plt.show()
            #plt.savefig("../plots/db_xyz.png")

        def plot3d(self):
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            for k, col in zip(self.unique_labels, colors):

                class_member_mask = (labels == k)
                if k != -1:
                    xy = X[class_member_mask & core_samples_mask]
                    ax.scatter(xy[:, 0], xy[:, 1], xy[:, 2], 'o', c=tuple(col),
                            s=160, label=k, edgecolors="grey")

                    xy = X[class_member_mask & ~core_samples_mask]
                    ax.scatter(xy[:, 0], xy[:, 1], xy[:, 2], 'o', c=tuple(col),
                            s=60, edgecolors="grey")
            #ax.scatter(0, 0, 0, marker="x", c="k")

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            plt.legend(loc=1)
            plt.show()

    def write_dtrajs(self,nframes):
        self.frames=nframes

        a_indexes = list(set(coordinates_and_labels[:, 1]))
        self.dtrajs = {}
        for a in a_indexes:
            dtrajs[a] = [-1] * nframes
            matches = coordinates_and_labels[self.coordinates_and_labels[:, 1] == a]
            for ind, params in enumerate(matches):
                self.dtrajs[a][int(params[0])] = params[-1]
        pickle.dump(self.dtrajs,open("dtrajs.p",'wb'),protocol=2)
        return self.dtrajs

    def plot_dtrajs(self,dtrajs):
        dtrajs=dtrajs

        plt.figure(figsize=(15, 7))
        for key in dtrajs.keys():
            if len(set(dtrajs[key])) >= 2:
                cls = list(set(dtrajs[key]))

                plt.scatter([i for i in range(0, self.frames, 1)], dtrajs[key], s=.5, label="Atom A index " + str(key))
                print("A atom with index nº " + str(key) + " visits clusters " + str(cls))
        plt.legend()
        plt.show()







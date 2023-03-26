import numpy as np
from Overlap import Overlap
from Dissimilarity import Dissimilarity

class DOC:
    """
    This class calculates the DOC matrix of a given cohort.
    """
    def __init__(self, cohort):
        """
        param cohort: a matrix, samples are in the rows.
        """
        self.cohort = cohort
        self.num_samples = np.size(cohort, 0)

    def calc_doc(self):
        """
        :return: matrix, the first row is the dissimilarity values for all the sample pairs of the
         cohort, the second row is for the overlap.
        """
        o = [[Overlap(self.cohort[j, :], self.cohort[i, :]).calculate_overlap() for i in range(
            j + 1, self.num_samples)] for j in range(0, self.num_samples - 1)]
        d = [[Dissimilarity(self.cohort[j, :], self.cohort[i, :]).calculate_dissimilarity() for i in range(
            j+1, self.num_samples)] for j in range(0, self.num_samples-1)]

        def flatten(lis):
            return [item for sublist in lis for item in sublist]

        o = np.array(flatten(o))
        d = np.array(flatten(d))
        doc_mat = np.vstack((o, d))

        return doc_mat

    def bootstrap(self):
        """
        :return: List of n DOC matrices after leave one out bootstrap is applied.
        """
        bootstrap_mat_container = []
        for i in range(0, self.num_samples):
            resampled_cohort = np.delete(self.cohort, i, axis=0)
            resampled_doc = DOC(resampled_cohort)
            bootstrap_mat_container.append(resampled_doc.calc_doc())
        return bootstrap_mat_container


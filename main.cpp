#include "Eigen/Eigenvalues"

#include <iostream>

using namespace Eigen;
using namespace std;

int main(int argc, char** argv)
{
	Matrix2d C;
	C <<
		5.0f, 5.0f,
		-1.0f, 7.0f;

	// Calculate SVD as described in https://www.youtube.com/watch?v=cOUTpqlX-Xs
	{
		Matrix2d CTC = C.transpose() * C;

		EigenSolver<Matrix2d> es(CTC);
		EigenSolver<Matrix2d>::EigenvalueType eigenValues = es.eigenvalues();
		EigenSolver<Matrix2d>::EigenvectorsType eigenVectors = es.eigenvectors();

		auto V = eigenVectors;

		MatrixXcd sigma;
		{
			sigma = MatrixXcd::Zero(eigenValues.rows(), eigenValues.rows());
			for (int i = 0; i < eigenValues.rows(); ++i)
				sigma(i, i) = sqrt(eigenValues[i].real());
		}

		MatrixXcd U = C * V;
		for (int i = 0; i < U.cols(); ++i)
			U.col(i) = U.col(i).normalized();

		cout << "U = \n" << U.real() << "\n\n";
		cout << "sigma = \n" << sigma.real() << "\n\n";
		cout << "V = \n" << V.real() << "\n\n";

		cout << "U * sigma * VT = \n" << (U * sigma * V.transpose()).real() << "\n\n";
		cout << "C = \n" << C << "\n\n";
	}

	/*
	MatrixXd A = MatrixXd::Random(2, 2);
	cout << "Here is a random 2x2 matrix, A:" << endl << A << endl << endl;

	EigenSolver<MatrixXd> es(A);
	cout << "The eigenvalues of A are:" << endl << es.eigenvalues() << endl;
	cout << "The matrix of eigenvectors, V, is:" << endl << es.eigenvectors() << endl << endl;

	complex<double> lambda = es.eigenvalues()[0];
	cout << "Consider the first eigenvalue, lambda = " << lambda << endl;
	VectorXcd v = es.eigenvectors().col(0);
	cout << "If v is the corresponding eigenvector, then lambda * v = " << endl << lambda * v << endl;
	cout << "... and A * v = " << endl << A.cast<complex<double> >() * v << endl << endl;

	MatrixXcd D = es.eigenvalues().asDiagonal();
	MatrixXcd V = es.eigenvectors();
	cout << "Finally, V * D * V^(-1) = " << endl << V * D * V.inverse() << endl;
	*/

	return 0;
}

/*

TODO:
- calculate SVD the way the video explains it
- then do it by the single function call!
- then do PCA too.

*/
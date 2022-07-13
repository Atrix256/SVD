#include "Eigen/Eigenvalues"
#include "Eigen/SVD"

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
			U.col(i).normalize();

		cout << "C = \n" << C << "\n\n";

		cout << "U = \n" << U.real() << "\n\n";
		cout << "sigma = \n" << sigma.real() << "\n\n";
		cout << "V = \n" << V.real() << "\n\n";

		cout << "U * sigma * VT = \n" << (U * sigma * V.transpose()).real() << "\n\n";
	}

	// Calculate SVD using the built in functionality
	{
		BDCSVD<Matrix2d> svd(C, ComputeFullU | ComputeFullV);

		auto U = svd.matrixU();
		auto V = svd.matrixV();
		auto sigma = svd.singularValues().asDiagonal().toDenseMatrix();

		cout << "C = \n" << C << "\n\n";

		cout << "U = \n" << U << "\n\n";
		cout << "sigma = \n" << sigma << "\n\n";
		cout << "V = \n" << V << "\n\n";

		cout << "U * sigma * VT = \n" << U * sigma * V.transpose() << "\n\n";
	}

	// PCA!
	{
		Matrix3d newC;
		newC <<
			0.002300, 0.043200, 0.002300,
			0.043200, 0.818000, 0.043200,
			0.002300, 0.043200, 0.002300;

		BDCSVD<Matrix3d> svd(newC, ComputeFullU | ComputeFullV);

		auto U = svd.matrixU();
		auto V = svd.matrixV();
		auto sigma = svd.singularValues().asDiagonal().toDenseMatrix();

		cout << "C = \n" << C << "\n\n";

		cout << "V = \n" << V << "\n\n";

		cout << "sigma = \n" << sigma << "\n\n";

		cout << "C * V = \n" << newC * V << "\n\n";

		cout << "Principle Direction = \n" << (newC * V).col(0).normalized() << "\n\n";
	}

	return 0;
}
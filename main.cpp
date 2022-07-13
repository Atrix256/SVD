#define _CRT_SECURE_NO_WARNINGS

#include "Eigen/Eigenvalues"
#include "Eigen/SVD"

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#include <vector>
#include <iostream>
#include <direct.h>

using namespace Eigen;
using namespace std;

void SaveMatrixResult(const char* fnprefix, const std::vector<double>& mtxpixels, int width, int height, int components, int percent)
{
	// rearrange the pixels and convert from double to unsigned char
	std::vector<unsigned char> pixels(width * height * components);

	const double* srcPixel = mtxpixels.data();
	for (size_t iy = 0; iy < height; ++iy)
	{
		for (size_t ic = 0; ic < components; ++ic)
		{
			for (size_t ix = 0; ix < width; ++ix)
			{
				pixels[(iy * width * components) + ix * components + ic] = (unsigned char)std::max(std::min(*srcPixel * 255.0, 255.0), 0.0);
				srcPixel++;
			}
		}
	}

	// write it out
	char fileName[1024];
	sprintf_s(fileName, "out/%s%i.png", fnprefix, percent);
	stbi_write_png(fileName, width, height, components, pixels.data(), 0);
}

int main(int argc, char** argv)
{
	_mkdir("out");

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

		cout << "C = \n" << newC << "\n\n";

		cout << "V = \n" << V << "\n\n";

		cout << "sigma = \n" << sigma << "\n\n";

		cout << "C * V = \n" << newC * V << "\n\n";

		cout << "Principle Direction = \n" << (newC * V).col(0).normalized() << "\n\n";
	}

	// Do truncated SVD on an image
	{
		// load the image
		int width, height, components;
		unsigned char* pixels = stbi_load("scenery.png", &width, &height, &components, 3);
		//unsigned char* pixels = stbi_load("lozman.png", &width, &height, &components, 3);
		components = 3;

		// convert it to double, and also give each component it's own row
		std::vector<double> mtxpixels(width * height * components);
		const unsigned char* srcPixel = pixels;
		for (size_t iy = 0; iy < height; ++iy)
		{
			for (size_t ix = 0; ix < width; ++ix)
			{
				for (size_t ic = 0; ic < components; ++ic)
				{
					mtxpixels[(iy * components + ic) * width + ix] = double(*srcPixel) / 255.0;
					srcPixel++;
				}
			}
		}
		stbi_image_free(pixels);

		// make the matrix
		Map<MatrixXd> mf(mtxpixels.data(), height * components, width);

		// Singular value decomposition
		BDCSVD<MatrixXd> svd(mf, ComputeFullU | ComputeFullV);
		auto U = svd.matrixU();
		auto V = svd.matrixV();
		auto singularValues = svd.singularValues();
		MatrixXd sigma = MatrixXd::Zero(U.cols(), singularValues.size());
		for (int svIndex = 0; svIndex < singularValues.size(); ++svIndex)
			sigma(svIndex, svIndex) = singularValues(svIndex);

		// Truncate the SVD to various percentages
		int percents[] =
		{
			100,
			75,
			50,
			25,
			20,
			15,
			10,
			5,
			0
		};

		for (int imageIndex = 0; imageIndex < _countof(percents); ++imageIndex)
		{
			// truncate
			int cutoffIndex = (int)sigma.cols() * percents[imageIndex] / 100;
			cutoffIndex = std::max(cutoffIndex, 1);
			for (int svIndex = cutoffIndex; svIndex < sigma.cols(); ++svIndex)
				sigma(svIndex, svIndex) = 0.0;

			// reconstruct the image
			mf = U * sigma * V.transpose();

			// save it out
			SaveMatrixResult("scenery", mtxpixels, width, height, components, percents[imageIndex]);
		}
	}

	return 0;
}
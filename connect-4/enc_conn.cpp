#include <iostream>
#include <fstream>
#include <string>
//#include <string.h>

using namespace std;

int main()
{
	std::ifstream infile;
	std::ofstream outfile;
	infile.open("conn.txt", ifstream::in);
	outfile.open("conn-enc.txt", ofstream::out);

int attrDims = 42;
while(!infile.eof())
{
	for(int i = 0; i < 42; i++)
	{
		char tempCh;
		infile >> tempCh;
		switch(tempCh)
		{
			case '1': 
				outfile << '1' << ' ' << '0' << ' ' << '0' << ' ';
				break;
			case '2':
				outfile << '0' << ' ' << '1' << ' ' << '0' << ' ';
				break;
			case '3':
				outfile << '0' << ' ' << '0' << ' ' << '1' << ' ';
				break;
			default:
				break;
		}
	}
	char label;
	infile >> label;
	outfile << label << '\n';
}

infile.close();
outfile.close();

return 0;
}


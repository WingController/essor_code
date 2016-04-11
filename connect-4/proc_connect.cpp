#include <iostream>
#include <fstream>
#include <string>
//#include <string.h>

using namespace std;

int main()
{
	std::ifstream infile;
	std::ofstream outfile;
	infile.open("connect-4.txt", ifstream::in);
	outfile.open("conn.txt", ofstream::out);

int attrDims = 42;
while(!infile.eof())
{
	for(int i = 0; i < 2*42; i++)
	{
		char tempCh;
		infile >> tempCh;
		switch(tempCh)
		{
			case ',': 
				outfile << ' ';
				break;
			case 'x':
				outfile << '1';
				break;
			case 'o':
				outfile << '2';
				break;
			case 'b':
				outfile << '3';
				break;
			default:
				break;
		}
	}

	// for the label col
/*	char label;
	infile >> label;
	
	switch(label)
	{
		case 'l':
			infile >> label >> label >> label;
			outfile << '1';
			break;
		case 'd':
			infile >> label >> label >> label;
			outfile << '2';
			break;
		case 'w':
			infile >> label >> label;
			outfile << '3';
			break;
		default:
			break;
	}
*/	

	string label;
	infile >> label;
	if(label.compare("loss") == 0)
	{
		outfile << '1';
	}
	else if(label.compare("draw") == 0)
	{
		outfile << '2';
	}
	else if(label.compare("win") == 0)
	{
		outfile << '3';
	}
	else
	{
		outfile << ' ';
	}
	outfile << '\n';

	//char* label = NULL;
	/*char label[4];
	infile.get(label,4,'\n');

	if(strcmp(label, "loss") == 0)
	{
		outfile << '1';
	}
	else if(strcmp(label, "draw") == 0)
	{
		outfile << '2';
	}
	else if(strcmp(label, "win") == 0)
	{
		outfile << '3';
	}
	else
	{
		outfile << ' ';
	}*/
	
}

infile.close();
outfile.close();

return 0;
}


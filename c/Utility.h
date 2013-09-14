#ifndef UTILITY_H
#define UTILITY_H

#include <string>
#include <Windows.h>

bool pathCreate(std::string path);
//HRESULT openFileDialog(std::string& txtFilePath);
bool fileExists(const char* filename);
// 

#endif
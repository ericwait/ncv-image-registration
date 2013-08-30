#include "Utility.h"
#include <Commctrl.h>
#include <shlobj.h>
#include <shobjidl.h> 
#include <shlwapi.h>
#include <shtypes.h>      // for COMDLG_FILTERSPEC
#include <fstream>

bool pathCreate(std::string path)
{
	size_t directoryEnd=path.find_first_of("/\\");

	while (directoryEnd<path.length()+1)
	{
		std::string dir = path.substr(0,directoryEnd);
		if (dir.find_last_of(".")!=std::string::npos)
			break;

		if (!CreateDirectoryA(dir.c_str(),NULL))
		{
			if (ERROR_PATH_NOT_FOUND==GetLastError())
				return false;
		}

		if (path.length()<=directoryEnd)
			break;

		std::string sub = path.substr(directoryEnd+1,std::string::npos);

		size_t temp = sub.find_first_of("/\\");
		if (temp==std::string::npos && path.length()>directoryEnd)
			directoryEnd = path.length();
		else
			directoryEnd += temp+1;
	}
	return true;
}

//#define INDEX_TXTFILE 1
//#if defined _M_IX86
//#pragma comment(linker, "/manifestdependency:\"type='win32' name='Microsoft.Windows.Common-Controls' version='6.0.0.0' processorArchitecture='x86' publicKeyToken='6595b64144ccf1df' language='*'\"")
//#elif defined _M_IA64
//#pragma comment(linker, "/manifestdependency:\"type='win32' name='Microsoft.Windows.Common-Controls' version='6.0.0.0' processorArchitecture='ia64' publicKeyToken='6595b64144ccf1df' language='*'\"")
//#elif defined _M_X64
//#pragma comment(linker, "/manifestdependency:\"type='win32' name='Microsoft.Windows.Common-Controls' version='6.0.0.0' processorArchitecture='amd64' publicKeyToken='6595b64144ccf1df' language='*'\"")
//#else
//#pragma comment(linker, "/manifestdependency:\"type='win32' name='Microsoft.Windows.Common-Controls' version='6.0.0.0' processorArchitecture='*' publicKeyToken='6595b64144ccf1df' language='*'\"")
//#endif
//
//IShellItemArray *psiResults = NULL;
//
//const COMDLG_FILTERSPEC c_rgSaveTypes[] =
//{
//	{L"Metadata File (*.txt)",       L"*.txt"},
//	{L"All Documents (*.*)",         L"*.*"}
//};
//
//class CDialogEventHandler : public IFileDialogEvents,
//	public IFileDialogControlEvents
//{
//public:
//	// IUnknown methods
//	IFACEMETHODIMP QueryInterface(REFIID riid, void** ppv)
//	{
//		static const QITAB qit[] = {
//			QITABENT(CDialogEventHandler, IFileDialogEvents),
//			QITABENT(CDialogEventHandler, IFileDialogControlEvents),
//			{ 0 },
//		};
//		return QISearch(this, qit, riid, ppv);
//	}
//
//	IFACEMETHODIMP_(ULONG) AddRef()
//	{
//		return InterlockedIncrement(&_cRef);
//	}
//
//	IFACEMETHODIMP_(ULONG) Release()
//	{
//		long cRef = InterlockedDecrement(&_cRef);
//		if (!cRef)
//			delete this;
//		return cRef;
//	}
//
//	// IFileDialogEvents methods
//	IFACEMETHODIMP OnFileOk(IFileDialog *) { return S_OK; };
//	IFACEMETHODIMP OnFolderChange(IFileDialog *) { return S_OK; };
//	IFACEMETHODIMP OnFolderChanging(IFileDialog *, IShellItem *) { return S_OK; };
//	IFACEMETHODIMP OnHelp(IFileDialog *) { return S_OK; };
//	IFACEMETHODIMP OnSelectionChange(IFileDialog *) { return S_OK; };
//	IFACEMETHODIMP OnShareViolation(IFileDialog *, IShellItem *, FDE_SHAREVIOLATION_RESPONSE *) { return S_OK; };
//	IFACEMETHODIMP OnTypeChange(IFileDialog *pfd) { return S_OK; };
//	IFACEMETHODIMP OnOverwrite(IFileDialog *, IShellItem *, FDE_OVERWRITE_RESPONSE *) { return S_OK; };
//
//	// IFileDialogControlEvents methods
//	IFACEMETHODIMP OnItemSelected(IFileDialogCustomize *pfdc, DWORD dwIDCtl, DWORD dwIDItem) { return S_OK; };
//	IFACEMETHODIMP OnButtonClicked(IFileDialogCustomize *, DWORD) { return S_OK; };
//	IFACEMETHODIMP OnCheckButtonToggled(IFileDialogCustomize *, DWORD, BOOL) { return S_OK; };
//	IFACEMETHODIMP OnControlActivating(IFileDialogCustomize *, DWORD) { return S_OK; };
//
//	CDialogEventHandler() : _cRef(1) { };
//private:
//	~CDialogEventHandler() { };
//	long _cRef;
//};
//
//// Instance creation helper
//HRESULT CDialogEventHandler_CreateInstance(REFIID riid, void **ppv)
//{
//	*ppv = NULL;
//	CDialogEventHandler *pDialogEventHandler = new (std::nothrow) CDialogEventHandler();
//	HRESULT hr = pDialogEventHandler ? S_OK : E_OUTOFMEMORY;
//	if (SUCCEEDED(hr))
//	{
//		hr = pDialogEventHandler->QueryInterface(riid, ppv);
//		pDialogEventHandler->Release();
//	}
//	return hr;
//}
//
//bool cvtLPW2stdstring(std::string& s, const LPWSTR pw, UINT codepage = 0U)
//{
//	bool res = false;
//	char* p = 0;
//	int bsz;
//
//	bsz = WideCharToMultiByte(codepage,
//		0,
//		pw,-1,
//		0,0,
//		0,0);
//	if (bsz > 0) {
//		p = new char[bsz];
//		int rc = WideCharToMultiByte(codepage,
//			0,
//			pw,-1,
//			p,bsz,
//			0,0);
//		if (rc != 0) {
//			p[bsz-1] = 0;
//			s = p;
//			res = true;
//		}
//	}
//	delete [] p;
//	return res;
//}
//
////#pragma optimize("",off)
//HRESULT openFileDialog(std::string& txtFilePath)
//{
//	IFileOpenDialog *pfd = NULL;
//	HRESULT hr = CoCreateInstance(CLSID_FileOpenDialog, NULL, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&pfd));
//	if (SUCCEEDED(hr))
//	{
//		// Set the options on the dialog.
//		DWORD dwFlags;
//
//		// Before setting, always get the options first in order not to override existing options.
//		hr = pfd->GetOptions(&dwFlags);
//		if (SUCCEEDED(hr))
//		{
//			// In this case, get shell items only for file system items.
//			hr = pfd->SetOptions(dwFlags | FOS_ALLOWMULTISELECT );
//			if (SUCCEEDED(hr))
//			{
//				// Set the file types to display only. Notice that, this is a 1-based array.
//				hr = pfd->SetFileTypes(ARRAYSIZE(c_rgSaveTypes), c_rgSaveTypes);
//				if (SUCCEEDED(hr))
//				{
//					// Set the selected file type index to txt for this example.
//					hr = pfd->SetFileTypeIndex(INDEX_TXTFILE);
//					if (SUCCEEDED(hr))
//					{
//						// Set the default extension to be ".txt" file.
//						hr = pfd->SetDefaultExtension(L"txt");
//						if (SUCCEEDED(hr))
//						{
//							// Show the dialog
//							hr = pfd->Show(NULL);
//							if (SUCCEEDED(hr))
//							{
//								// Obtain the result, once the user clicks the 'Open' button.
//								// The result is an IShellItem object.
//								IShellItem *psiResult;
//								hr = pfd->GetResults(&psiResults);
//								if (SUCCEEDED(hr))
//								{
//									DWORD count;
//									hr = psiResults->GetCount(&count);
//									if (SUCCEEDED(hr))
//									{
//										LPWSTR filePath;
//										if (1==count)
//										{
//											hr = psiResults->GetItemAt(0,&psiResult);
//											if (SUCCEEDED(hr))
//											{
//												hr = psiResult->GetDisplayName(SIGDN_FILESYSPATH,&filePath);
//												if(SUCCEEDED(hr))
//												{
//													if(cvtLPW2stdstring(txtFilePath,filePath))
//													{
//														return S_OK;
//													}else
//														return S_FALSE;
//												}
//											}
//										}
//									}
//								}
//							}
//						}
//					}
//				}
//			}
//		}
//	}
//	return hr;
//}

 bool fileExists(const char* filename){
 	std::ifstream ifile(filename);
 	bool rtn = ifile.good();
 	ifile.close();
 	return rtn;
 }

#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <memory>
#include <algorithm>
#include <cassert>

// Called if any error occurs during context creation or at kernel runtime.
// This can be called many times asynchronously, so it must be thread safe.
void errorCallback(const char *ErrorInfo, const void *PrivateInfo, size_t CB, void *UserData)
{
    std::cerr << "Error found (callback):\n" << ErrorInfo << "\n";
    std::cerr.flush();
}

using PlatformDevicePair = std::pair<cl::Platform, cl::Device>;

#if EMBEDDED
    using ResultType = cl_uint;
#else
    using ResultType = cl_ulong;
#endif

class CLLauncherArguments
{
    public:
        std::unique_ptr<std::string> ExecutableName;
        std::unique_ptr<std::string> KernelFile;
        std::unique_ptr<std::string> ArgsFile;
        std::unique_ptr<unsigned> DeviceIdx;
        std::unique_ptr<std::string> DeviceName;
        std::unique_ptr<unsigned> PlatformIdx;
        std::unique_ptr<size_t> BinarySize;
        std::unique_ptr<cl::NDRange> LocalWS;
        std::unique_ptr<cl::NDRange> GlobalWS;
        std::unique_ptr<std::string> IncludePath;
        std::unique_ptr<unsigned> AtomicsNum;
        bool UseAtomicReductions{false};
        bool UseFakeDivergence{false};
        bool UseInterThreadCommunication{false};
        bool UseEMI{false};
        bool SetDeviceFromName{false};
        bool DebugMode{false};
        bool OutputBinary{false};
        bool OptDisable{false};
        bool DisableFakeDivergence{false};
        bool DisableGroupDivergence{false};
        bool DisableAtomics{false};
};

void printHelp(const std::string ProgramName)
{
    std::cout << "Usage: " << ProgramName << " -f <cl_program> -p <platform_idx> -d <device_idx> [flags...]\n";
    std::cout << "\n";
    std::cout << "Required flags are:\n";
    std::cout << "  -f FILE --filename FILE                   Test file\n";
    std::cout << "  -p IDX  --platform_idx IDX                Target platform\n";
    std::cout << "  -d IDX  --device_idx IDX                  Target device\n";
    std::cout << "\n";
    std::cout << "Optional flags are:\n";
    std::cout << "  -i PATH --include_path PATH               Include path for kernels (. by default)\n"; //FGG
    std::cout << "  -b N    --binary N                        Compiles the kernel to binary, allocating N bytes\n";
    std::cout << "  -l N    --locals N                        A string with comma-separated values representing the number of work-units per group per dimension\n";
    std::cout << "  -g N    --groups N                        Same as -l, but representing the total number of work-units per dimension\n";
    std::cout << "  -n NAME --name NAME                       Ensure the device name contains this string\n";
    std::cout << "  -a FILE --args FILE                       Look for file-defined arguments in this file, rather than the test file\n";
    std::cout << "          --atomics                         Test uses atomic sections\n";
    std::cout << "                      ---atomic_reductions  Test uses atomic reductions\n";
    std::cout << "                      ---emi                Test uses EMI\n";
    std::cout << "                      ---fake_divergence    Test uses fake divergence\n";
    std::cout << "                      ---inter_thread_comm  Test uses inter-thread communication\n";
    std::cout << "                      ---debug              Print debug info\n";
    std::cout << "                      ---bin                Output disassembly of kernel in out.bin\n";
    std::cout << "                      ---disable_opts       Disable OpenCL compile optimisations\n";
    std::cout << "                      ---disable_group      Disable group divergence feature\n";
    std::cout << "                      ---disable_fake       Disable fake divergence feature\n";
    std::cout << "                      ---disable_atomics    Disable atomic sections and reductions\n";
    std::cout << "                      ---set_device_from_name\n";
    std::cout << "                                            Ignore target platform -p and device -d\n";
    std::cout << "                                            Instead try to find a matching platform/device based on the device name\n";
}

std::unique_ptr<cl::NDRange> stringToNDRange(const std::string &Val)
{
    std::array<unsigned, 3> WS;
    unsigned Dimension = 0;
    auto I = 0;
    auto Pos = Val.find(",");

    while(Pos != std::string::npos)
    {
		WS[Dimension] = stoi(Val.substr(I, Pos - I));
        I = ++Pos;
        ++Dimension;
        Pos = Val.find(",", Pos);
    }

    switch(Dimension)
    {
        case 0:
            return std::make_unique<cl::NDRange>();
        case 1:
            return std::make_unique<cl::NDRange>(WS[0]);
        case 2:
            return std::make_unique<cl::NDRange>(WS[0], WS[1]);
        case 3:
            return std::make_unique<cl::NDRange>(WS[0], WS[1], WS[2]);
        default:
            return nullptr;
    }
}

std::unique_ptr<PlatformDevicePair> getPlatformDeviceConfig(const CLLauncherArguments &Args)
{
    assert(Args.PlatformIdx && "Invalid platform index!");
    assert(Args.DeviceIdx && "Invalid device index!");

    std::vector<cl::Platform> Platforms;
    std::vector<cl::Device> Devices;

    cl::Platform::get(&Platforms);

    if(*Args.PlatformIdx >= Platforms.size())
    {
        std::cerr << "No platform for index " << *Args.PlatformIdx << "!\n";
        return nullptr;
    }

    auto Platform = Platforms[*Args.PlatformIdx];

    Platform.getDevices(CL_DEVICE_TYPE_ALL, &Devices);

    if(*Args.DeviceIdx >= Devices.size())
    {
        std::cerr << "No device for index " << *Args.DeviceIdx << "!\n";
        return nullptr;
    }

    auto Device = Devices[*Args.DeviceIdx];
    auto DeviceName = Device.getInfo<CL_DEVICE_NAME>();

    if(Args.DeviceName && DeviceName.find(*Args.DeviceName) == std::string::npos)
    {
        std::cerr << "Given name, " << *Args.DeviceName << ", not found in device name, " << DeviceName << "!\n";
        return nullptr;
    }

    return std::make_unique<PlatformDevicePair>(Platform, Device);
}

bool setDeviceFromDeviceName(CLLauncherArguments &Args)
{
    assert(Args.DeviceName);

    std::vector<cl::Platform> Platforms;
    cl::Platform::get(&Platforms);
    unsigned PlatformIdx = 0;

    for(auto &Platform : Platforms)
    {
        unsigned DeviceIdx = 0;
        std::vector<cl::Device> Devices;

        Platform.getDevices(CL_DEVICE_TYPE_ALL, &Devices);

        for(auto &Device : Devices)
        {
            std::string DeviceName = Device.getInfo<CL_DEVICE_NAME>();

            if(DeviceName.find(*Args.DeviceName) != std::string::npos)
            {
                Args.PlatformIdx = std::make_unique<unsigned>(PlatformIdx);
                Args.DeviceIdx = std::make_unique<unsigned>(DeviceIdx);

                return true;
            }

            ++DeviceIdx;
        }

        ++PlatformIdx;
    }

    return false;
}

void parseArg(const std::string &Arg, const std::string &Val, CLLauncherArguments &Args)
{
    if(Arg == "-h" || Arg == "--help")
    {
        printHelp(*Args.ExecutableName);
        exit(0);
    }
    else if(Arg == "-f" || Arg == "--filename")
    {
        Args.KernelFile = std::make_unique<std::string>(Val);
    }
    else if(Arg == "-a" || Arg == "--args")
    {
        Args.ArgsFile = std::make_unique<std::string>(Val);
    }
    else if(Arg == "-d" || Arg == "--device_idx")
    {
        Args.DeviceIdx = std::make_unique<unsigned>(stoi(Val));
    }
    else if(Arg == "-p" || Arg == "--platform_idx")
    {
        Args.PlatformIdx = std::make_unique<unsigned>(stoi(Val));
    }
    else if(Arg == "-b" || Arg == "--binary")
    {
        Args.BinarySize = std::make_unique<size_t>(stoi(Val));
    }
    else if(Arg == "-l" || Arg == "--locals")
    {
        Args.LocalWS = stringToNDRange(Val);
    }
    else if(Arg == "-g" || Arg == "--groups")
    {
        Args.GlobalWS = stringToNDRange(Val);
    }
    else if(Arg == "-n" || Arg == "--name")
    {
        Args.DeviceName = std::make_unique<std::string>(Val);
    }
    else if(Arg == "-i" || Arg == "--include_path")
    {
        auto IncludePath = std::make_unique<std::string>(Val);
        size_t Pos;
        unsigned Offset = 0;

        while((Pos = IncludePath->find("\\", Offset)) != std::string::npos)
        {
            IncludePath->replace(Pos, 1, "/");
            Offset = Pos + 1;
        }

        Args.IncludePath = std::move(IncludePath);
    }
    else if(Arg == "--atomics")
    {
        Args.AtomicsNum = std::make_unique<unsigned>(stoi(Val));
    }
    else if(Arg == "---set_device_from_name")
    {
        Args.SetDeviceFromName = true;
    }
    else if(Arg == "---atomic_reductions")
    {
        std::cout << "AtomicReductions\n";
        Args.UseAtomicReductions = true;
    }
    else if(Arg == "---emi")
    {
        Args.UseEMI = true;
    }
    else if(Arg == "---fake_divergence")
    {
        Args.UseFakeDivergence = true;
    }
    else if(Arg == "---inter_thread_comm")
    {
        Args.UseInterThreadCommunication = true;
    }
    else if(Arg == "---debug")
    {
        Args.DebugMode = true;
    }
    else if(Arg == "---bin")
    {
        Args.OutputBinary = true;
    }
    else if(Arg == "---disable_opts")
    {
        Args.OptDisable = true;
    }
    else if(Arg == "---disable_fake")
    {
        Args.DisableFakeDivergence = true;
    }
    else if(Arg == "---disable_group")
    {
        Args.DisableGroupDivergence = true;
    }
    else if(Arg == "---disable_atomics")
    {
        Args.DisableAtomics = true;
    }
    else
    {
        std::cerr << "Failed parsing Arg " << Arg << ".\n";
    }
}

void parseCommandlineArgs(int argc, char** argv, CLLauncherArguments &Args)
{
    for(int ArgIdx = 1; ArgIdx < argc; ++ArgIdx)
    {
        std::string CurrArg(argv[ArgIdx]);
        std::string NextArg = "";

        // Arguments with 3 dashes don't have values
        if(CurrArg.find("---") != 0)
        {
            if(++ArgIdx >= argc)
            {
                std::cerr << "Found option " << CurrArg << " with no value.\n";
                continue;
            }

            NextArg = std::string(argv[ArgIdx]);
        }

        parseArg(CurrArg, NextArg, Args);
    }
}

void parseFileArgs(CLLauncherArguments &Args)
{
    std::ifstream File;

	if(Args.ArgsFile)
	{
        File.open(*Args.ArgsFile);
    }
    else if(Args.KernelFile)
    {
        File.open(*Args.KernelFile);
    }
    else
    {
        return;
    }

    if(!File.good())
    {
        return;
    }

    std::string Line;
    std::getline(File, Line);
    File.close();

    if(Line.find("//") == 0)
    {
		auto I = 0;
		auto Pos = Line.find(" ");

		while(Pos != std::string::npos)
		{
			const std::string &Arg = Line.substr(I, Pos - I);
			I = ++Pos;

            if(Arg.find("---") == 0)
            {
                parseArg(Arg, "", Args);
            }
            else if(Arg.find("-") == 0)
            {
				Pos = Line.find(" ", Pos);
				const std::string &Val = Line.substr(I, Pos - I);

                parseArg(Arg, Val, Args);

                if(Pos == std::string::npos)
                {
                    break;
                }

				I = ++Pos;
            }

			Pos = Line.find(" ", Pos);
		}
    }
}

CLLauncherArguments parseArguments(int argc, char** argv)
{
    CLLauncherArguments Args;

    Args.ExecutableName = std::make_unique<std::string>(argv[0]);

    // Parsing commandline arguments (1st run)
    parseCommandlineArgs(argc, argv, Args);

    // Parse arguments found in the given files
    parseFileArgs(Args);

    // Parse commandline arguments (2nd run)
    parseCommandlineArgs(argc, argv, Args);

    if(Args.SetDeviceFromName)
    {
        if(!Args.DeviceName)
        {
            std::cerr << "Must give '-n NAME' to use --set_device_from_name!\n";
        }
        else
        {
            bool Success = setDeviceFromDeviceName(Args);

            if(!Success)
            {
                std::cerr << "No matching device found for name " << *Args.DeviceName << "!\n";
            }
        }
    }

    return Args;
}

bool isSaneConfig(const CLLauncherArguments &Args)
{
    if(!Args.KernelFile)
    {
        std::cerr << "Require file (-f) argument!\n";
        return false;
    }

    if((!Args.DeviceIdx || !Args.PlatformIdx) && !Args.DeviceName)
    {
        std::cerr << "Require device index (-d) and platform index (-p) arguments, or device name (-n)!\n";
        return false;
    }

    if(!Args.GlobalWS)
    {
        std::cerr << "Invalid global work sizes! Maximum is three dimensions.\n";
        return false;
    }

    if(!Args.LocalWS)
    {
        std::cerr << "Invalid local work sizes! Maximum is three dimensions.\n";
        return false;
    }

    if(Args.GlobalWS->dimensions() != Args.LocalWS->dimensions())
    {
        std::cerr << "Local and global sizes must have same number of dimensions!\n";
        return false;
    }

    for(unsigned Dim = 0; Dim < Args.GlobalWS->dimensions(); ++Dim)
    {
        if((*Args.LocalWS)[Dim] > (*Args.GlobalWS)[Dim])
        {
            std::cerr << "Local work size in dimension " << Dim << " greater than global work size!\n";
            return false;
        }
    }

    if(!Args.PlatformIdx || !Args.DeviceIdx)
    {
        return false;
    }

    return true;
}

std::string createBuildOptions(const CLLauncherArguments &Args)
{
    std::ostringstream BuildOptions;

    BuildOptions << "-w";

    if(Args.IncludePath)
    {
        BuildOptions << " -I" << *Args.IncludePath;
    }
    else
    {
        BuildOptions << " -I.";
    }

    if(Args.OptDisable)
    {
        BuildOptions << " -cl-opt-disable";
    }

    if(Args.DisableGroupDivergence)
    {
        BuildOptions << " -DNO_GROUP_DIVERGENCE";
    }

    if(Args.DisableFakeDivergence)
    {
        BuildOptions << " -DNO_FAKE_DIVERGENCE";
    }

    if(Args.DisableAtomics)
    {
        BuildOptions << " -DNO_ATOMICS";
    }

    return BuildOptions.str();
}

unsigned calculateTotalWorkItemNum(const CLLauncherArguments &Args)
{
    if(Args.GlobalWS->dimensions() == 0)
    {
        return 0;
    }

    unsigned TotalWorkItemNum = 1;

    for(unsigned Dim = 0; Dim < Args.GlobalWS->dimensions(); ++Dim)
    {
        TotalWorkItemNum *= (*Args.GlobalWS)[Dim];
    }

    return TotalWorkItemNum;
}

unsigned calculateTotalWorkGroupNum(const CLLauncherArguments &Args)
{
    if(Args.GlobalWS->dimensions() == 0)
    {
        return 0;
    }

    unsigned TotalWorkGroupNum = 1;

    for(unsigned Dim = 0; Dim < Args.GlobalWS->dimensions(); ++Dim)
    {
        TotalWorkGroupNum *= (*Args.GlobalWS)[Dim] / (*Args.LocalWS)[Dim];
    }

    return TotalWorkGroupNum;
}

int main(int argc, char** argv)
{
    // Parse the input. Expect three parameters.
    if(argc < 4)
    {
        std::cerr << "Expected at least three arguments!\n";
        printHelp(argv[0]);
        return 1;
    }

    CLLauncherArguments Args = parseArguments(argc, argv);

    if(!isSaneConfig(Args))
    {
        return 1;
    }

    try
    {
        auto PlatformDeviceConfig = getPlatformDeviceConfig(Args);

        if(!PlatformDeviceConfig)
        {
            return 1;
        }

        cl_context_properties ContextProperties[] = {
            CL_CONTEXT_PLATFORM,
            (cl_context_properties)(PlatformDeviceConfig->first()),
            0
        };

        cl::Context Context(PlatformDeviceConfig->second, ContextProperties, errorCallback, nullptr);
        cl::CommandQueue CommandQueue(Context, PlatformDeviceConfig->second);

        std::vector<cl::Device> Devices{PlatformDeviceConfig->second};

        std::unique_ptr<cl::Program> Program = nullptr;

        std::ifstream SourceFile(*Args.KernelFile, std::ios::in | std::ios::binary);

        if(!SourceFile.good())
        {
            std::cerr << "Failed to load kernel file " << *Args.KernelFile << "!\n";
            return 1;
        }

        if(Args.BinarySize)
        {
            auto Buffer = std::unique_ptr<char[]>(new char[*Args.BinarySize]);
            SourceFile.read(Buffer.get(), *Args.BinarySize);

            cl::Program::Binaries Binaries(1, std::make_pair(Buffer.get(), *Args.BinarySize));
            //TODO: Does this work? Or is Binaries not retained?
            Program = std::make_unique<cl::Program>(Context, Devices, std::move(Binaries));
        }
        else
        {
            std::ostringstream SourceStream;
            SourceStream << SourceFile.rdbuf();

            Program = std::make_unique<cl::Program>(Context, SourceStream.str());
        }

        SourceFile.close();
        std::string BuildOptions = createBuildOptions(Args);

        const auto BuildError = Program->build(Devices, BuildOptions.c_str());

        if(BuildError != CL_SUCCESS)
        {
            std::cerr << "Error building program!\n";

            if(Args.DebugMode)
            {
                auto BuildInfo = Program->getBuildInfo<CL_PROGRAM_BUILD_LOG>(PlatformDeviceConfig->second);
                std::cerr << BuildInfo;
            }
        }

        std::cerr << "Compilation terminated successfully..." << std::endl;

        if(Args.OutputBinary)
        {
            auto Binaries = Program->getInfo<CL_PROGRAM_BINARIES>();

            std::ofstream BinaryFile("out.bin", std::ios::out | std::ios::binary);
            BinaryFile << Binaries[0];
            BinaryFile.close();
        }

        auto Kernel = cl::Kernel(*Program, "entry");

        const unsigned TotalWorkItemNum = calculateTotalWorkItemNum(Args);
        const unsigned TotalWorkGroupNum = calculateTotalWorkGroupNum(Args);

        ResultType *ResultHostMemory = nullptr;

        cl::Buffer ResultBuffer(Context, CL_MEM_WRITE_ONLY, TotalWorkItemNum * sizeof(ResultType));

        ResultHostMemory = static_cast<ResultType*>(CommandQueue.enqueueMapBuffer(ResultBuffer, CL_TRUE, CL_MAP_WRITE_INVALIDATE_REGION, 0, TotalWorkItemNum * sizeof(ResultType)));
        std::memset(ResultHostMemory, 0, TotalWorkItemNum * sizeof(ResultType));
        CommandQueue.enqueueUnmapMemObject(ResultBuffer, ResultHostMemory);

        unsigned KernelArg = 0;
        Kernel.setArg(KernelArg++, ResultBuffer);

        std::unique_ptr<cl_uint[]> AtomicValuesHostMemory = nullptr;
        std::unique_ptr<cl::Buffer> AtomicValuesBuffer = nullptr;
        std::unique_ptr<cl_uint[]> SpecialValuesHostMemory = nullptr;
        std::unique_ptr<cl::Buffer> SpecialValuesBuffer = nullptr;

        if(Args.AtomicsNum)
        {
            const unsigned BufferSize = TotalWorkGroupNum * *Args.AtomicsNum;

            AtomicValuesHostMemory = std::make_unique<cl_uint[]>(BufferSize);
            std::memset(AtomicValuesHostMemory.get(), 0, BufferSize * sizeof(cl_uint));

            AtomicValuesBuffer = std::make_unique<cl::Buffer>(Context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, BufferSize * sizeof(cl_uint), AtomicValuesHostMemory.get());
            
            Kernel.setArg(KernelArg++, sizeof(cl_mem), AtomicValuesBuffer.get());
            
            SpecialValuesHostMemory = std::make_unique<cl_uint[]>(BufferSize);
            std::memset(AtomicValuesHostMemory.get(), 0, BufferSize * sizeof(cl_uint));

            SpecialValuesBuffer = std::make_unique<cl::Buffer>(Context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, BufferSize * sizeof(cl_uint), SpecialValuesHostMemory.get());
            
            Kernel.setArg(KernelArg++, sizeof(cl_mem), SpecialValuesBuffer.get());
        }

        std::unique_ptr<cl_int[]> AtomicReductionValuesHostMemory = nullptr;
        std::unique_ptr<cl::Buffer> AtomicReductionValuesBuffer = nullptr;

        if(Args.UseAtomicReductions)
        {
            AtomicReductionValuesHostMemory = std::make_unique<cl_int[]>(TotalWorkGroupNum);
            std::memset(AtomicReductionValuesHostMemory.get(), 0, TotalWorkGroupNum * sizeof(cl_int));

            AtomicReductionValuesBuffer = std::make_unique<cl::Buffer>(Context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, TotalWorkGroupNum * sizeof(cl_int), AtomicReductionValuesHostMemory.get());
            
            Kernel.setArg(KernelArg++, sizeof(cl_mem), AtomicReductionValuesBuffer.get());
        }

        std::unique_ptr<cl_int[]> EMIValuesHostMemory = nullptr;
        std::unique_ptr<cl::Buffer> EMIValuesBuffer = nullptr;

        if(Args.UseEMI)
        {
            const unsigned BufferSize = 1024;

            EMIValuesHostMemory = std::make_unique<cl_int[]>(BufferSize);

            for(unsigned I = 0; I < BufferSize; ++I)
            {
                EMIValuesHostMemory[I] = BufferSize - I;
            }

            EMIValuesBuffer = std::make_unique<cl::Buffer>(Context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, BufferSize * sizeof(cl_int), EMIValuesHostMemory.get());
            
            Kernel.setArg(KernelArg++, sizeof(cl_mem), EMIValuesBuffer.get());
        }

        std::unique_ptr<cl_int[]> FakeDivergenceValuesHostMemory = nullptr;
        std::unique_ptr<cl::Buffer> FakeDivergenceValuesBuffer = nullptr;

        if(Args.UseFakeDivergence)
        {
            unsigned BufferSize = 0;

            for(unsigned Dim = 0; Dim < Args.GlobalWS->dimensions(); ++Dim)
            {
                if((*Args.GlobalWS)[Dim] > BufferSize)
                {
                    BufferSize = (*Args.GlobalWS)[Dim];
                }
            }

            FakeDivergenceValuesHostMemory = std::make_unique<cl_int[]>(BufferSize);

            for(unsigned I = 0; I < BufferSize; ++I)
            {
                FakeDivergenceValuesHostMemory[I] = 10 + I;
            }

            FakeDivergenceValuesBuffer = std::make_unique<cl::Buffer>(Context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, BufferSize * sizeof(cl_int), FakeDivergenceValuesHostMemory.get());
            
            Kernel.setArg(KernelArg++, sizeof(cl_mem), FakeDivergenceValuesBuffer.get());
        }

        std::unique_ptr<cl_long[]> InterThreadCommValuesHostMemory = nullptr;
        std::unique_ptr<cl::Buffer> InterThreadCommValuesBuffer = nullptr;

        if(Args.UseInterThreadCommunication)
        {
            InterThreadCommValuesHostMemory = std::make_unique<cl_long[]>(TotalWorkItemNum);

            for(unsigned I = 0; I < TotalWorkItemNum; ++I)
            {
                InterThreadCommValuesHostMemory[I] = 1;
            }

            InterThreadCommValuesBuffer = std::make_unique<cl::Buffer>(Context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, TotalWorkItemNum * sizeof(cl_long), InterThreadCommValuesHostMemory.get());
            
            Kernel.setArg(KernelArg++, sizeof(cl_mem), InterThreadCommValuesBuffer.get());
        }

        CommandQueue.enqueueNDRangeKernel(Kernel, cl::NullRange, *Args.GlobalWS, *Args.LocalWS);
        CommandQueue.finish();

        ResultHostMemory = static_cast<ResultType*>(CommandQueue.enqueueMapBuffer(ResultBuffer, CL_TRUE, CL_MAP_READ, 0, TotalWorkItemNum * sizeof(ResultType)));

        std::cout << std::hex;

        for(unsigned I = 0; I < TotalWorkItemNum; ++I)
        {
            std::cout << "0x" << ResultHostMemory[I] << ",";
        }

        std::cout << std::dec << std::endl;

        CommandQueue.enqueueUnmapMemObject(ResultBuffer, ResultHostMemory);
    }
    catch (cl::Error Error)
    {
        std::cerr << "ERROR: " << Error.what() << "(" << Error.err() << ")" << std::endl;
    }
}

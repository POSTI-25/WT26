#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>

#include <curl/curl.h>
#include <nvml.h>
#include <nlohmann/json.hpp>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

using json = nlohmann::json;

namespace {

constexpr const char* kServerUrl = "http://localhost:5000/update";
constexpr int kIntervalSeconds = 3;
constexpr unsigned int kGpuIndex = 0;

struct GpuStats {
    double gpuUtil = 0.0;
    unsigned long long memUsed = 0;
    unsigned long long memTotal = 0;
    double memUsedPct = 0.0;
    double freeMemMb = 0.0;
    unsigned int activeProcesses = 0;
    unsigned int temperature = 0;
};

std::string getHostname() {
#ifdef _WIN32
    char name[MAX_COMPUTERNAME_LENGTH + 1] = {0};
    DWORD size = MAX_COMPUTERNAME_LENGTH + 1;
    if (GetComputerNameA(name, &size)) {
        return std::string(name);
    }
#else
    char name[256] = {0};
    if (gethostname(name, sizeof(name)) == 0) {
        return std::string(name);
    }
#endif
    return "unknown-node";
}

bool getGpuStats(GpuStats& stats, std::string& error) {
    nvmlDevice_t device;

    nvmlReturn_t rc = nvmlDeviceGetHandleByIndex(kGpuIndex, &device);
    if (rc != NVML_SUCCESS) {
        error = std::string("nvmlDeviceGetHandleByIndex failed: ") + nvmlErrorString(rc);
        return false;
    }

    nvmlUtilization_t util{};
    rc = nvmlDeviceGetUtilizationRates(device, &util);
    if (rc != NVML_SUCCESS) {
        error = std::string("nvmlDeviceGetUtilizationRates failed: ") + nvmlErrorString(rc);
        return false;
    }

    nvmlMemory_t mem{};
    rc = nvmlDeviceGetMemoryInfo(device, &mem);
    if (rc != NVML_SUCCESS) {
        error = std::string("nvmlDeviceGetMemoryInfo failed: ") + nvmlErrorString(rc);
        return false;
    }

    unsigned int temp = 0;
    rc = nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temp);
    if (rc != NVML_SUCCESS) {
        error = std::string("nvmlDeviceGetTemperature failed: ") + nvmlErrorString(rc);
        return false;
    }

    unsigned int processCount = 0;
    rc = nvmlDeviceGetComputeRunningProcesses(device, &processCount, nullptr);
    if (rc != NVML_SUCCESS && rc != NVML_ERROR_INSUFFICIENT_SIZE) {
        if (rc == NVML_ERROR_NOT_SUPPORTED) {
            processCount = 0;
        } else {
            error = std::string("nvmlDeviceGetComputeRunningProcesses failed: ") + nvmlErrorString(rc);
            return false;
        }
    }

    stats.gpuUtil = static_cast<double>(util.gpu);
    stats.memUsed = mem.used;
    stats.memTotal = mem.total;
    stats.memUsedPct = (mem.total == 0)
        ? 0.0
        : (static_cast<double>(mem.used) / static_cast<double>(mem.total)) * 100.0;
    stats.freeMemMb = static_cast<double>(mem.free) / (1024.0 * 1024.0);
    stats.activeProcesses = processCount;
    stats.temperature = temp;

    return true;
}

bool evaluateAvailability(const GpuStats& stats) {
    return stats.activeProcesses == 0 &&
           stats.memUsedPct < 50.0 &&
           stats.gpuUtil < 50.0 &&
           stats.freeMemMb >= 2000.0 &&
           stats.temperature < 85;
}

bool sendToServer(const json& payload, const std::string& url) {
    CURL* curl = curl_easy_init();
    if (!curl) {
        std::cerr << "Failed to initialize CURL" << std::endl;
        return false;
    }

    std::string payloadStr = payload.dump();
    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_POST, 1L);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payloadStr.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, static_cast<long>(payloadStr.size()));
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 5L);

    CURLcode curlRc = curl_easy_perform(curl);
    if (curlRc != CURLE_OK) {
        std::cerr << "HTTP request error: " << curl_easy_strerror(curlRc) << std::endl;
        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);
        return false;
    }

    long httpCode = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &httpCode);
    std::cout << "Server response: " << httpCode << std::endl;

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    return (httpCode >= 200 && httpCode < 300);
}

}  // namespace

int main() {
    const std::string nodeId = getHostname();

    CURLcode globalCurlRc = curl_global_init(CURL_GLOBAL_DEFAULT);
    if (globalCurlRc != CURLE_OK) {
        std::cerr << "curl_global_init failed: " << curl_easy_strerror(globalCurlRc) << std::endl;
        return 1;
    }

    while (true) {
        nvmlReturn_t initRc = nvmlInit();
        if (initRc != NVML_SUCCESS) {
            std::cerr << "NVML init failed: " << nvmlErrorString(initRc)
                      << ". Retrying next cycle..." << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(kIntervalSeconds));
            continue;
        }

        unsigned int gpuCount = 0;
        nvmlReturn_t countRc = nvmlDeviceGetCount(&gpuCount);
        if (countRc != NVML_SUCCESS) {
            std::cerr << "nvmlDeviceGetCount failed: " << nvmlErrorString(countRc) << std::endl;
            nvmlShutdown();
            std::this_thread::sleep_for(std::chrono::seconds(kIntervalSeconds));
            continue;
        }

        if (gpuCount <= kGpuIndex) {
            std::cerr << "No compatible GPU found. Retrying next cycle..." << std::endl;
            nvmlShutdown();
            std::this_thread::sleep_for(std::chrono::seconds(kIntervalSeconds));
            continue;
        }

        GpuStats stats;
        std::string error;
        bool statsOk = getGpuStats(stats, error);

        if (!statsOk) {
            std::cerr << "GPU stats error: " << error << std::endl;
            nvmlShutdown();
            std::this_thread::sleep_for(std::chrono::seconds(kIntervalSeconds));
            continue;
        }

        bool available = evaluateAvailability(stats);
        json payload = {
            {"node_id", nodeId},
            {"gpu_util", stats.gpuUtil},
            {"mem_used_pct", stats.memUsedPct},
            {"free_mem_mb", stats.freeMemMb},
            {"active_processes", stats.activeProcesses},
            {"temperature", stats.temperature},
            {"available", available}
        };

        std::cout << "Sending data from node " << nodeId << "..." << std::endl;
        if (!available) {
            std::cout << "GPU not available" << std::endl;
        }

        sendToServer(payload, kServerUrl);

        nvmlReturn_t shutdownRc = nvmlShutdown();
        if (shutdownRc != NVML_SUCCESS) {
            std::cerr << "NVML shutdown warning: " << nvmlErrorString(shutdownRc) << std::endl;
        }

        std::this_thread::sleep_for(std::chrono::seconds(kIntervalSeconds));
    }

    curl_global_cleanup();
    return 0;
}

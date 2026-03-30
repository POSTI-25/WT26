#include <filesystem>
#include <fstream>
#include <grpcpp/grpcpp.h>
#include <iostream>
#include <string>
#include <vector>

#include "transfer.grpc.pb.h"

int main(int argc, char **argv) {
    // Usage: sender <receiver_host> <file_path> [port] [chunk_size_bytes]
    if (argc < 3) {
        std::cerr << "Usage: sender <receiver_host> <file_path> [port] [chunk_size_bytes]\n";
        return 1;
    }

    const std::string receiver_host = argv[1];
    const std::string file_path = argv[2];
    const std::string port = (argc > 3) ? argv[3] : "50051";
    const std::size_t chunk_size = (argc > 4) ? static_cast<std::size_t>(std::stoull(argv[4]))
                                               : 1024 * 1024;

    if (!std::filesystem::exists(file_path)) {
        std::cerr << "[sender] File not found: " << file_path << "\n";
        return 1;
    }

    const std::uint64_t file_size =
        static_cast<std::uint64_t>(std::filesystem::file_size(file_path));
    const std::string file_name = std::filesystem::path(file_path).filename().string();

    std::ifstream in(file_path, std::ios::binary);
    if (!in) {
        std::cerr << "[sender] Failed to open file: " << file_path << "\n";
        return 1;
    }

    const std::string target = receiver_host + ":" + port;
    auto channel = grpc::CreateChannel(target, grpc::InsecureChannelCredentials());
    auto stub = transfer::FileTransferService::NewStub(channel);

    grpc::ClientContext context;
    transfer::UploadReply reply;
    auto writer = stub->UploadFile(&context, &reply);

    transfer::UploadRequest meta_request;
    transfer::FileMetadata *meta = meta_request.mutable_meta();
    meta->set_filename(file_name);
    meta->set_total_size(file_size);
    meta->set_chunk_size(static_cast<std::uint32_t>(chunk_size));
    if (!writer->Write(meta_request)) {
        std::cerr << "[sender] Failed to send metadata.\n";
        return 1;
    }

    std::vector<char> buffer(chunk_size, 0);
    std::uint64_t sent = 0;

    while (in) {
        in.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const std::streamsize n = in.gcount();
        if (n <= 0) {
            break;
        }

        transfer::UploadRequest data_request;
        data_request.set_data(buffer.data(), static_cast<std::size_t>(n));
        if (!writer->Write(data_request)) {
            std::cerr << "\n[sender] Stream closed by receiver while sending data.\n";
            return 1;
        }

        sent += static_cast<std::uint64_t>(n);
        std::cout << "\r[sender] Progress: " << sent << "/" << file_size << " bytes"
                  << std::flush;
    }

    writer->WritesDone();
    grpc::Status status = writer->Finish();

    if (!status.ok()) {
        std::cerr << "\n[sender] Upload failed: " << status.error_message() << "\n";
        return 1;
    }

    if (!reply.ok()) {
        std::cerr << "\n[sender] Receiver reported failure: " << reply.message() << "\n";
        return 1;
    }

    std::cout << "\n[sender] Transfer complete. Receiver stored " << reply.bytes_received()
              << " bytes.\n";
    return 0;
}

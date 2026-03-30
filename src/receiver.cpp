#include <cstdint>
#include <filesystem>
#include <fstream>
#include <grpcpp/grpcpp.h>
#include <iostream>
#include <string>

#include "transfer.grpc.pb.h"

class FileTransferServiceImpl final : public transfer::FileTransferService::Service {
  public:
    explicit FileTransferServiceImpl(std::string output_dir)
        : output_dir_(std::move(output_dir)) {}

    grpc::Status UploadFile(grpc::ServerContext *context,
                            grpc::ServerReader<transfer::UploadRequest> *reader,
                            transfer::UploadReply *reply) override {
        (void)context;

        std::filesystem::create_directories(output_dir_);

        transfer::UploadRequest request;
        if (!reader->Read(&request) || !request.has_meta()) {
            reply->set_ok(false);
            reply->set_message("First stream message must be metadata.");
            reply->set_bytes_received(0);
            return grpc::Status::OK;
        }

        const transfer::FileMetadata &meta = request.meta();
        const std::string safe_name = std::filesystem::path(meta.filename()).filename().string();
        const std::uint64_t expected_size = meta.total_size();
        const std::filesystem::path out_path =
            std::filesystem::path(output_dir_) / ("received_" + safe_name);

        std::ofstream out(out_path, std::ios::binary);
        if (!out) {
            reply->set_ok(false);
            reply->set_message("Failed to open output file.");
            reply->set_bytes_received(0);
            return grpc::Status::OK;
        }

        std::uint64_t written = 0;
        while (reader->Read(&request)) {
            if (!request.has_data()) {
                continue;
            }
            const std::string &chunk = request.data();
            out.write(chunk.data(), static_cast<std::streamsize>(chunk.size()));
            written += static_cast<std::uint64_t>(chunk.size());
            std::cout << "\r[receiver] Progress: " << written << "/" << expected_size
                      << " bytes" << std::flush;
        }

        out.close();
        std::cout << "\n[receiver] Saved file to " << out_path << "\n";

        if (written != expected_size) {
            reply->set_ok(false);
            reply->set_message("Size mismatch: wrote " + std::to_string(written) +
                               " expected " + std::to_string(expected_size));
            reply->set_bytes_received(written);
            return grpc::Status::OK;
        }

        reply->set_ok(true);
        reply->set_message("Transfer complete.");
        reply->set_bytes_received(written);
        return grpc::Status::OK;
    }

  private:
    std::string output_dir_;
};

int main(int argc, char **argv) {
    // Usage: receiver [port] [output_dir]
    const std::string port = (argc > 1) ? argv[1] : "50051";
    const std::string output_dir = (argc > 2) ? argv[2] : ".\\incoming";
    const std::string server_addr = "0.0.0.0:" + port;

    FileTransferServiceImpl service(output_dir);

    grpc::ServerBuilder builder;
    builder.AddListeningPort(server_addr, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    std::unique_ptr<grpc::Server> server(builder.BuildAndStart());

    if (!server) {
        std::cerr << "[receiver] Failed to start gRPC server on " << server_addr << "\n";
        return 1;
    }

    std::cout << "[receiver] gRPC server listening on " << server_addr << "\n";
    server->Wait();
    return 0;
}

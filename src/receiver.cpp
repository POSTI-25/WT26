#include <zmq.hpp>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

static std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        out.push_back(item);
    }
    return out;
}

int main(int argc, char **argv) {
    // Usage: receiver [control_port] [data_port] [output_dir]
    const std::string control_port = (argc > 1) ? argv[1] : "5555";
    const std::string data_port = (argc > 2) ? argv[2] : "6000";
    const std::string output_dir = (argc > 3) ? argv[3] : ".\\incoming";

    std::filesystem::create_directories(output_dir);

    zmq::context_t ctx(1);

    zmq::socket_t rep(ctx, zmq::socket_type::rep);
    rep.bind("tcp://0.0.0.0:" + control_port);

    std::cout << "[receiver] Waiting metadata on port " << control_port << "...\n";
    zmq::message_t meta_msg;
    if (!rep.recv(meta_msg, zmq::recv_flags::none)) {
        std::cerr << "[receiver] Failed to receive metadata.\n";
        return 1;
    }

    std::string meta(static_cast<char *>(meta_msg.data()), meta_msg.size());
    std::vector<std::string> parts = split(meta, '|');
    if (parts.size() != 4 || parts[0] != "META") {
        rep.send(zmq::buffer("ERR_BAD_META"), zmq::send_flags::none);
        std::cerr << "[receiver] Invalid metadata format.\n";
        return 1;
    }

    const std::string safe_name = std::filesystem::path(parts[1]).filename().string();
    const std::uint64_t expected_size = std::stoull(parts[2]);
    const std::uint64_t chunk_size = std::stoull(parts[3]);

    std::filesystem::path out_path =
        std::filesystem::path(output_dir) / ("received_" + safe_name);
    std::ofstream out(out_path, std::ios::binary);
    if (!out) {
        rep.send(zmq::buffer("ERR_OPEN_FILE"), zmq::send_flags::none);
        std::cerr << "[receiver] Failed to open output file: " << out_path << "\n";
        return 1;
    }

    rep.send(zmq::buffer("OK"), zmq::send_flags::none);
    std::cout << "[receiver] Receiving " << safe_name << " (" << expected_size
              << " bytes), chunk size " << chunk_size << " bytes.\n";

    zmq::socket_t pull(ctx, zmq::socket_type::pull);
    pull.bind("tcp://0.0.0.0:" + data_port);

    std::uint64_t written = 0;
    while (written < expected_size) {
        zmq::message_t type_msg;
        zmq::message_t seq_msg;
        zmq::message_t data_msg;

        if (!pull.recv(type_msg, zmq::recv_flags::none)) {
            std::cerr << "\n[receiver] Failed to receive frame type.\n";
            return 1;
        }

        std::string type(static_cast<char *>(type_msg.data()), type_msg.size());
        if (type == "DATA") {
            if (!pull.recv(seq_msg, zmq::recv_flags::none) ||
                !pull.recv(data_msg, zmq::recv_flags::none)) {
                std::cerr << "\n[receiver] Failed to receive DATA frames.\n";
                return 1;
            }

            out.write(static_cast<char *>(data_msg.data()),
                      static_cast<std::streamsize>(data_msg.size()));
            written += static_cast<std::uint64_t>(data_msg.size());
            std::cout << "\r[receiver] Progress: " << written << "/" << expected_size
                      << " bytes" << std::flush;
        } else if (type == "DONE") {
            if (!pull.recv(seq_msg, zmq::recv_flags::none)) {
                std::cerr << "\n[receiver] Failed to receive DONE sequence.\n";
                return 1;
            }
            break;
        } else {
            std::cerr << "\n[receiver] Unknown frame type: " << type << "\n";
            return 1;
        }
    }

    out.close();
    std::cout << "\n[receiver] Saved file to " << out_path << "\n";

    if (written != expected_size) {
        std::cerr << "[receiver] Warning: size mismatch, wrote " << written
                  << " but expected " << expected_size << "\n";
        return 2;
    }

    std::cout << "[receiver] Transfer complete.\n";
    return 0;
}

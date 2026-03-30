#include <zmq.hpp>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char **argv) {
    // Usage: sender <receiver_ip> <file_path> [control_port] [data_port] [chunk_size_bytes]
    if (argc < 3) {
        std::cerr << "Usage: sender <receiver_ip> <file_path> [control_port] [data_port] "
                     "[chunk_size_bytes]\n";
        return 1;
    }

    const std::string receiver_ip = argv[1];
    const std::string file_path = argv[2];
    const std::string control_port = (argc > 3) ? argv[3] : "5555";
    const std::string data_port = (argc > 4) ? argv[4] : "6000";
    const std::size_t chunk_size =
        (argc > 5) ? static_cast<std::size_t>(std::stoull(argv[5])) : 1024 * 1024;

    if (!std::filesystem::exists(file_path)) {
        std::cerr << "[sender] File not found: " << file_path << "\n";
        return 1;
    }

    const std::uint64_t file_size = std::filesystem::file_size(file_path);
    const std::string file_name = std::filesystem::path(file_path).filename().string();

    std::ifstream in(file_path, std::ios::binary);
    if (!in) {
        std::cerr << "[sender] Failed to open file: " << file_path << "\n";
        return 1;
    }

    zmq::context_t ctx(1);

    zmq::socket_t req(ctx, zmq::socket_type::req);
    req.connect("tcp://" + receiver_ip + ":" + control_port);

    const std::string meta = "META|" + file_name + "|" + std::to_string(file_size) + "|" +
                             std::to_string(chunk_size);
    req.send(zmq::buffer(meta), zmq::send_flags::none);

    zmq::message_t rep_msg;
    if (!req.recv(rep_msg, zmq::recv_flags::none)) {
        std::cerr << "[sender] No response from receiver on control channel.\n";
        return 1;
    }

    std::string rep(static_cast<char *>(rep_msg.data()), rep_msg.size());
    if (rep != "OK") {
        std::cerr << "[sender] Receiver rejected metadata: " << rep << "\n";
        return 1;
    }

    zmq::socket_t push(ctx, zmq::socket_type::push);
    push.connect("tcp://" + receiver_ip + ":" + data_port);

    std::vector<char> buffer(chunk_size);
    std::uint64_t sent = 0;
    std::uint64_t seq = 0;

    while (in) {
        in.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const std::streamsize n = in.gcount();
        if (n <= 0) {
            break;
        }

        push.send(zmq::buffer("DATA"), zmq::send_flags::sndmore);
        const std::string seq_s = std::to_string(seq++);
        push.send(zmq::buffer(seq_s), zmq::send_flags::sndmore);
        push.send(zmq::buffer(buffer.data(), static_cast<std::size_t>(n)),
                  zmq::send_flags::none);

        sent += static_cast<std::uint64_t>(n);
        std::cout << "\r[sender] Progress: " << sent << "/" << file_size << " bytes"
                  << std::flush;
    }

    push.send(zmq::buffer("DONE"), zmq::send_flags::sndmore);
    const std::string final_seq = std::to_string(seq);
    push.send(zmq::buffer(final_seq), zmq::send_flags::none);

    std::cout << "\n[sender] Transfer complete.\n";
    return 0;
}

// GD AI Bridge
//
// Opens a TCP server on 127.0.0.1:22222. A Python agent connects and drives the game in
// lockstep: the game only advances when the agent asks for a step, and every step lasts
// exactly 1/60 s of game time (= 4 physics ticks at 240 Hz). This makes the game
// deterministic and lets it run much faster than real time.
//
// Protocol (one text line per message, "\n"-terminated):
//   agent -> game                game -> agent
//   STEP 0 | STEP 1              S x y vy ground dead won percent mode
//   RESET                        S ...   (state right after the restart)
//   STATE                        S ...   (current state, no time passes)
//   SPEED n                      OK      (n steps per rendered frame, 1 = real time)
//
// When no agent is connected, the game behaves exactly as usual.

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#endif

#include <Geode/Geode.hpp>
#include <Geode/modify/CCScheduler.hpp>
#include <Geode/modify/PlayLayer.hpp>

#include <algorithm>
#include <cstdlib>
#include <string>

using namespace geode::prelude;

namespace bridge {

constexpr u_short kPort = 22222;
constexpr float kStepDt = 1.f / 60.f;     // one agent decision = 1/60 s of game time
constexpr DWORD kRecvTimeoutMs = 50;      // keep the window responsive while the agent thinks

SOCKET listenSock = INVALID_SOCKET;
SOCKET client = INVALID_SOCKET;
std::string inbox;
int stepsPerFrame = 20;                   // game steps per rendered frame (speedhack)
bool holding = false;                     // is the jump button currently held by the agent
bool won = false;

bool connected() {
    return client != INVALID_SOCKET;
}

void startServer() {
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
        log::error("WSAStartup failed");
        return;
    }
    listenSock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (listenSock == INVALID_SOCKET) {
        log::error("socket() failed");
        return;
    }
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(kPort);
    inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
    if (bind(listenSock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == SOCKET_ERROR ||
        listen(listenSock, 1) == SOCKET_ERROR) {
        log::error("Cannot listen on 127.0.0.1:{}", kPort);
        closesocket(listenSock);
        listenSock = INVALID_SOCKET;
        return;
    }
    u_long nonBlocking = 1;
    ioctlsocket(listenSock, FIONBIO, &nonBlocking);   // accept() must never freeze the game
    log::info("AI bridge listening on 127.0.0.1:{}", kPort);
}

void releaseButton(PlayLayer* pl) {
    if (pl && holding) {
        pl->handleButton(false, 1, true);
    }
    holding = false;
}

void dropClient(PlayLayer* pl) {
    releaseButton(pl);
    closesocket(client);
    client = INVALID_SOCKET;
    inbox.clear();
    log::info("AI client disconnected");
}

void tryAccept() {
    if (listenSock == INVALID_SOCKET || connected()) return;
    SOCKET s = accept(listenSock, nullptr, nullptr);
    if (s == INVALID_SOCKET) return;                  // nobody is waiting

    u_long blocking = 0;
    ioctlsocket(s, FIONBIO, &blocking);
    DWORD timeout = kRecvTimeoutMs;
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<char const*>(&timeout), sizeof(timeout));
    BOOL noDelay = TRUE;                              // send small messages immediately
    setsockopt(s, IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<char const*>(&noDelay), sizeof(noDelay));

    client = s;
    inbox.clear();
    holding = false;
    won = false;
    log::info("AI client connected");
}

enum class ReadResult { Line, Timeout, Closed };

ReadResult readLine(std::string& line) {
    while (true) {
        auto pos = inbox.find('\n');
        if (pos != std::string::npos) {
            line = inbox.substr(0, pos);
            inbox.erase(0, pos + 1);
            if (!line.empty() && line.back() == '\r') line.pop_back();
            return ReadResult::Line;
        }
        char buf[512];
        int n = recv(client, buf, sizeof(buf), 0);
        if (n > 0) {
            inbox.append(buf, n);
            continue;
        }
        if (n == SOCKET_ERROR && WSAGetLastError() == WSAETIMEDOUT) return ReadResult::Timeout;
        return ReadResult::Closed;
    }
}

bool sendLine(std::string const& s) {
    size_t sent = 0;
    while (sent < s.size()) {
        int n = send(client, s.data() + sent, static_cast<int>(s.size() - sent), 0);
        if (n == SOCKET_ERROR) return false;
        sent += n;
    }
    return true;
}

int gamemode(PlayerObject* p) {
    if (p->m_isShip) return 1;
    if (p->m_isBall) return 2;
    if (p->m_isBird) return 3;      // UFO
    if (p->m_isDart) return 4;      // wave
    if (p->m_isRobot) return 5;
    if (p->m_isSpider) return 6;
    if (p->m_isSwing) return 7;
    return 0;                       // cube
}

std::string stateLine(PlayLayer* pl) {
    auto p = pl->m_player1;
    bool hasWon = won || pl->m_levelEndAnimationStarted;
    return fmt::format("S {:.4f} {:.4f} {:.4f} {} {} {} {:.3f} {}\n",
        p->m_position.x, p->m_position.y, p->m_yVelocity,
        static_cast<int>(p->m_isOnGround), static_cast<int>(p->m_isDead), static_cast<int>(hasWon),
        pl->getCurrentPercent(), gamemode(p));
}

} // namespace bridge

$on_mod(Loaded) {
    bridge::startServer();
}

class $modify(BridgeScheduler, CCScheduler) {
    void update(float dt) {
        using namespace bridge;
        tryAccept();

        auto pl = PlayLayer::get();
        if (!connected() || !pl || pl->m_isPaused) {
            CCScheduler::update(dt);          // no agent: normal game
            return;
        }

        // Lockstep: answer the agent's commands until we have done `stepsPerFrame` steps,
        // then let GD render one frame.
        int steps = 0;
        while (steps < stepsPerFrame) {
            std::string line;
            auto res = readLine(line);
            if (res == ReadResult::Timeout) return;          // agent busy: the game waits for it
            if (res == ReadResult::Closed) {
                dropClient(pl);
                CCScheduler::update(dt);
                return;
            }

            std::string reply;
            if (line.rfind("STEP", 0) == 0) {
                bool press = line.size() > 5 && line[5] == '1';
                if (press != holding) {
                    pl->handleButton(press, 1, true);         // button 1 = jump, player 1
                    holding = press;
                }
                CCScheduler::update(kStepDt);                 // advance exactly 1/60 s
                pl = PlayLayer::get();
                if (!pl) return;                              // level was left
                ++steps;
                reply = stateLine(pl);
            }
            else if (line == "RESET") {
                releaseButton(pl);
                won = false;
                pl->resetLevel();
                reply = stateLine(pl);
            }
            else if (line == "STATE") {
                reply = stateLine(pl);
            }
            else if (line.rfind("SPEED", 0) == 0) {
                stepsPerFrame = std::clamp(std::atoi(line.c_str() + 5), 1, 1000);
                reply = "OK\n";
            }
            else {
                reply = "E unknown command\n";
            }

            if (!sendLine(reply)) {
                dropClient(pl);
                return;
            }
        }
    }
};

class $modify(BridgePlayLayer, PlayLayer) {
    // After a death GD schedules its own restart about 1 s later. The agent restarts the
    // level itself, so this delayed restart would hit in the middle of the next attempt.
    void delayedResetLevel() {
        if (bridge::connected()) return;
        PlayLayer::delayedResetLevel();
    }

    // Do not open the end screen while the agent is playing: just remember the win.
    void levelComplete() {
        if (bridge::connected()) {
            bridge::won = true;
            return;
        }
        PlayLayer::levelComplete();
    }
};

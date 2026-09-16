using System;
using System.Collections.Concurrent;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using Celeste.Mod;

namespace CelesteRL.Lockstep;

/// A message from one connection. The generation identifies which connection sent it.
internal readonly record struct IncomingMessage(int Generation, string Text);

/// Localhost TCP server speaking newline-delimited JSON with one Python client at a time.
///
/// Every accepted connection gets a new generation number. A new connection replaces the current one,
/// so a stuck client cannot block recovery. Messages carry the generation they arrived on, and replies
/// are only written if their generation is still current, so one connection can never receive another
/// connection's results.
internal sealed class LockstepServer : IDisposable {
    private const string LogTag = "CelesteRLLockstep";
    // A client that stops reading must not block the game thread indefinitely on a write.
    private const int SendTimeoutMs = 2000;

    private readonly TcpListener listener;
    private readonly Thread acceptThread;
    private readonly BlockingCollection<IncomingMessage> incoming = new();
    private readonly object connectionLock = new();
    private volatile bool disposed;

    private TcpClient? client;
    private StreamWriter? writer;
    private int generation;

    public LockstepServer(int port) {
        // Loopback only: nothing outside this machine can drive the game.
        listener = new TcpListener(IPAddress.Loopback, port);
        acceptThread = new Thread(AcceptLoop) { IsBackground = true, Name = "CelesteRL lockstep accept" };
    }

    /// Generation of the connected client, or 0 if none is connected.
    public int CurrentGeneration {
        get {
            lock (connectionLock) {
                return client != null ? generation : 0;
            }
        }
    }

    public void Start() {
        listener.Start();
        acceptThread.Start();
    }

    /// Wait up to `timeout` for a message from connection `forGeneration`, the connection the caller is
    /// serving. Messages from older connections are discarded. A message from a newer connection is held
    /// back and returned once the caller asks for that generation, so it is never handled as if it came
    /// from the connection it replaced. Called only from the game thread.
    public bool TryTake(int forGeneration, out string message, TimeSpan timeout) {
        message = "";
        if (held is { } waiting) {
            if (waiting.Generation > forGeneration) {
                return false;
            }
            held = null;
            if (waiting.Generation == forGeneration) {
                message = waiting.Text;
                return true;
            }
        }

        var deadline = DateTime.UtcNow + timeout;
        while (true) {
            var remaining = deadline - DateTime.UtcNow;
            if (remaining < TimeSpan.Zero || !incoming.TryTake(out var item, remaining)) {
                return false;
            }
            if (item.Generation == forGeneration) {
                message = item.Text;
                return true;
            }
            if (item.Generation > forGeneration) {
                held = item;
                return false;
            }
        }
    }

    // Only touched by TryTake, which only the game thread calls.
    private IncomingMessage? held;

    /// Write a reply if `forGeneration` is still the connected client. Returns false if it was dropped.
    public bool Send(int forGeneration, string line) {
        lock (connectionLock) {
            if (client == null || writer == null || forGeneration != generation) {
                return false;
            }
            try {
                writer.Write(line);
                writer.Write('\n');
                writer.Flush();
                return true;
            } catch (IOException) {
                Logger.Log(LogLevel.Warn, LogTag, $"Write to connection {generation} failed or timed out; closing it");
                CloseCurrent();
                return false;
            } catch (ObjectDisposedException) {
                CloseCurrent();
                return false;
            }
        }
    }

    private void AcceptLoop() {
        while (!disposed) {
            TcpClient accepted;
            try {
                accepted = listener.AcceptTcpClient();
            } catch (SocketException) {
                return; // listener stopped
            } catch (ObjectDisposedException) {
                return;
            }

            // Nagle's algorithm would delay each small reply by tens of milliseconds.
            accepted.NoDelay = true;
            accepted.SendTimeout = SendTimeoutMs;

            int thisGeneration;
            lock (connectionLock) {
                CloseCurrent();
                generation += 1;
                thisGeneration = generation;
                client = accepted;
                writer = new StreamWriter(accepted.GetStream(), new UTF8Encoding(false)) { AutoFlush = false };
            }
            Logger.Log(LogLevel.Info, LogTag, $"Python client connected (connection {thisGeneration})");

            var reader = new Thread(() => ReadLoop(accepted, thisGeneration)) {
                IsBackground = true,
                Name = $"CelesteRL lockstep reader {thisGeneration}",
            };
            reader.Start();
        }
    }

    private void ReadLoop(TcpClient connection, int connectionGeneration) {
        try {
            using var reader = new StreamReader(connection.GetStream(), new UTF8Encoding(false));
            while (reader.ReadLine() is { } line) {
                incoming.Add(new IncomingMessage(connectionGeneration, line));
            }
        } catch (IOException) {
        } catch (ObjectDisposedException) {
        } catch (InvalidOperationException) {
        }

        lock (connectionLock) {
            if (client == connection) {
                CloseCurrent();
            }
        }
        Logger.Log(LogLevel.Info, LogTag, $"Python client disconnected (connection {connectionGeneration})");
    }

    /// Caller must hold connectionLock.
    private void CloseCurrent() {
        writer = null;
        client?.Close();
        client = null;
    }

    public void Dispose() {
        disposed = true;
        listener.Stop();
        lock (connectionLock) {
            CloseCurrent();
        }
    }
}

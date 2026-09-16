using System;
using System.Collections.Concurrent;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using Celeste.Mod;

namespace CelesteRL.Lockstep;

/// Localhost TCP server for one Python client at a time, speaking newline-delimited JSON.
/// Messages are read on a background thread and handed to the game thread through a queue;
/// replies are written from the game thread.
internal sealed class LockstepServer : IDisposable {
    private readonly TcpListener listener;
    private readonly Thread acceptThread;
    private readonly object writeLock = new();
    private volatile bool disposed;

    private TcpClient? client;
    private StreamWriter? writer;
    private BlockingCollection<string> incoming = new();

    public LockstepServer(int port) {
        // Loopback only: nothing outside this machine can drive the game.
        listener = new TcpListener(IPAddress.Loopback, port);
        acceptThread = new Thread(AcceptLoop) { IsBackground = true, Name = "CelesteRL lockstep accept" };
    }

    public bool Connected => client?.Connected == true;

    public void Start() {
        listener.Start();
        acceptThread.Start();
    }

    /// Wait up to `timeout` for the next message. Returns false if none arrived.
    public bool TryTake(out string message, TimeSpan timeout) {
        return incoming.TryTake(out message!, timeout);
    }

    public void Send(string line) {
        lock (writeLock) {
            try {
                writer?.Write(line);
                writer?.Write('\n');
                writer?.Flush();
            } catch (IOException) {
                Disconnect();
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
            }

            // Nagle's algorithm would delay each small reply by tens of milliseconds.
            accepted.NoDelay = true;
            lock (writeLock) {
                Disconnect();
                incoming = new BlockingCollection<string>();
                client = accepted;
                writer = new StreamWriter(accepted.GetStream(), new UTF8Encoding(false)) { AutoFlush = false };
            }
            Logger.Log(LogLevel.Info, "CelesteRLLockstep", "Python client connected");

            try {
                using var reader = new StreamReader(accepted.GetStream(), new UTF8Encoding(false));
                while (reader.ReadLine() is { } line) {
                    incoming.Add(line);
                }
            } catch (IOException) {
            } catch (ObjectDisposedException) {
            }

            lock (writeLock) {
                if (client == accepted) {
                    Disconnect();
                }
            }
            Logger.Log(LogLevel.Info, "CelesteRLLockstep", "Python client disconnected");
        }
    }

    private void Disconnect() {
        writer = null;
        client?.Close();
        client = null;
    }

    public void Dispose() {
        disposed = true;
        listener.Stop();
        lock (writeLock) {
            Disconnect();
        }
    }
}

import "package:flutter/foundation.dart";

enum WalletConnectionState
{
    NONE,
    CONNECTED,
    DISCONNECTED,
    ERROR,
}

class WalletProvider with ChangeNotifier
{
    WalletConnectionState _connectionState = WalletConnectionState.NONE;
    double _balance = 0;
    String _lastError = "";
    String _receiveAddress = "";

    bool get isConnected => _connectionState == WalletConnectionState.CONNECTED;
    WalletConnectionState get connectionState => _connectionState;
    double get balance => _balance;
    String get receiveAddress => _receiveAddress;

    WalletProvider()
    {
        initializeStreams();

        // TODO: Stub
        // (() async
        // {
        //     await Future.delayed(Duration(seconds: 1));
        //     _connectionState = WalletConnectionState.CONNECTED;
        //     _receiveAddress = "pyrintest:qpr6vyju9ftrc5hcer6vwd59mjhnl4efwgrekmfhuyp6qqs55eq7udsftcesr";
        //     _balance = 15e10;
        //     notifyListeners();
        // })();
    }

    void initializeStreams()
    {

    }

    void connect()
    {
    }

    void disconnect()
    {
        // Simulating wallet disconnection
        _connectionState = WalletConnectionState.DISCONNECTED;
        _balance = 0;
        notifyListeners();
    }

    void updateBalance(double newBalance)
    {
        _balance = newBalance;
        notifyListeners();
    }
}

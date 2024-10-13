import "dart:convert";

import "package:flutter/material.dart";
import "package:flutter/services.dart";
import "package:provider/provider.dart";
import "package:pyrin_app/assets_section.dart";
import "package:pyrin_app/components/navigation_bar.dart";
import "package:pyrin_app/core/wallet_provider.dart";
import "package:pyrin_app/core/wasm.dart";
import "package:pyrin_app/home_section.dart";
import "package:pyrin_app/menu_section.dart";
import "package:pyrin_app/receive_page.dart";
import "package:pyrin_app/send_page.dart";
import "package:pyrin_app/swap_section.dart";
import "package:pyrin_app/token/token_creation.dart";
import "package:pyrin_app/tokens_page.dart";
import "package:pyrin_app/ui.dart";
import "package:pyrin_app/wallet/create_wallet.dart";
import "package:pyrin_app/wallet/enter_wallet_password.dart";
import "package:pyrin_app/wallet/import_wallet.dart";
import "package:pyrin_app/wallet/setup_wallet_biometric.dart";
import "package:pyrin_app/wallet/setup_wallet_password.dart";
import "package:pyrin_app/welcome/welcome_page.dart";
// import "package:wasm_run_flutter/wasm_run_flutter.dart";

import "package:pyrin/pyrin.dart";


void main() async
{
    runApp(ChangeNotifierProvider(
      create: (context) => WalletProvider(),
      child: const App(),
    ));

    WidgetsFlutterBinding.ensureInitialized();

    SystemChrome.setSystemUIOverlayStyle(const SystemUiOverlayStyle(
      statusBarColor: Colors.transparent,
    ));

    final pyrin = Pyrin();
    final result = pyrin.add(5, 3);
    print('5 + 3 = $result');

    // final result2 = await pyrin.connect();
    final result2 = pyrin.connect();
    print('result2 = $result2');

    // await init();
}

class App extends StatefulWidget
{
    const App({super.key});

    @override
    State<App> createState() => _AppState();
}

class _AppState extends State<App>
{
    Section _section = Section.HOME;

    @override
    Widget build(BuildContext context)
    {
        final routes =
        {
            // "/": (context) => HomePage(),
            "/send": (context) => const SendPage(),
            "/receive": (context) => const ReceivePage(),
            "/tokens": (context) => TokensPage(),

            // Wallet
            "/wallet/create": (context) => const CreateWalletPage(),
            "/wallet/import": (context) => const ImportWalletPage(),
            "/wallet/password/enter": (context) => const EnterWalletPasswordPage(),
            "/wallet/password/setup": (context) => const SetupWalletPasswordPage(),
            "/wallet/biometric/setup": (context) => const SetupWalletBiometricPage(),

            // Token
            "/token/create": (context) => const TokenCreatePage(),

            // Welcome
            "/welcome": (context) => WelcomePage(),
        };

        return MaterialApp(
          // initialRoute: "/",
          // initialRoute: kDebugMode ? "/welcome" : "/",
          initialRoute: "/",
          routes: routes,
          // onGenerateRoute: onGenerateRoute,
          theme: ThemeData(
            colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
            scaffoldBackgroundColor: PyrinColors.BACKGROUND_COLOR,
            textSelectionTheme: TextSelectionThemeData(
              cursorColor: PyrinColors.TEXT_COLOR,
              selectionColor: PyrinColors.TEXT_COLOR.withOpacity(0.3),
              selectionHandleColor: PyrinColors.TEXT_COLOR,
            ),
            textTheme: TextTheme(
              bodySmall: TextStyle(
                color: PyrinColors.TEXT_COLOR,
                fontFamily: "Inter",
              ),
              bodyMedium: TextStyle(
                color: PyrinColors.TEXT_COLOR,
                fontFamily: "Inter",
              ),
              bodyLarge: TextStyle(
                color: PyrinColors.TEXT_COLOR,
                fontFamily: "Inter",
              ),
            ),
            useMaterial3: true,
          ),
          home: Scaffold(
            body: Center(
              child: SafeArea(
                child: Stack(
                  children: [
                    _buildSection(),
                    Align(
                      alignment: Alignment.bottomCenter,
                      child: PyrinNavigationBar(
                        section: _section,
                        onSectionChanged: (section) => setState(() => _section = section),
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),
        );
    }

    Widget _buildSection()
    {
        switch (_section)
        {
            case Section.HOME:
                return HomeSection();
            case Section.SWAP:
                return SwapSection();
            case Section.ASSETS:
                return AssetsSection();
            case Section.MENU:
                return MenuSection();
        }
    }
}

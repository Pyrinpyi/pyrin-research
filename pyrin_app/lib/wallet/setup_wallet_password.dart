import "package:flutter/material.dart";
import "package:pyrin_app/core/page.dart";
import "package:pyrin_app/ui.dart";

class SetupWalletPasswordPage extends StatefulWidget
{
  const SetupWalletPasswordPage({super.key});

  @override
  State<SetupWalletPasswordPage> createState() => _SetupWalletPasswordState();
}

class _SetupWalletPasswordState extends State<SetupWalletPasswordPage>
{
    @override
    Widget build(BuildContext context)
    {
        return RoutePage(
          name: "Password",
          buttons: [
            PyrinElevatedButton(
              text: "Next",
              onClick: onNextClick,
              wide: true,
            )
          ],
          child: Column(
            children: [
              PyrinTitle(text: "Set up your password"),
              PyrinSubtitle(text: "Setup a strong password to secure your wallet."),

              Container(
                margin: const EdgeInsets.only(bottom: 20),
                child: PyrinPasswordTextField(
                  name: "Password",
                  hintText: "Enter your password",
                ),
              ),
              Container(
                margin: const EdgeInsets.only(bottom: 20),
                child: PyrinPasswordTextField(
                  name: "Confirm Password",
                  hintText: "Confirm your password",
                ),
              ),
              const SizedBox(height: 40),
              PyrinCardGroup(
                name: "Enable Biometric",
                text: "Enable biometric authentication for quick access to your wallet.",
                leading: PyrinSwitch(),
              ),
            ],
          ),
        );
    }

    onNextClick()
    {

    }
}

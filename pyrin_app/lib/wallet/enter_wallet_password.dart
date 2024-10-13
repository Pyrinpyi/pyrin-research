import "package:flutter/material.dart";
import "package:pyrin_app/core/page.dart";
import "package:pyrin_app/ui.dart";

class EnterWalletPasswordPage extends StatefulWidget
{
  const EnterWalletPasswordPage({super.key});

  @override
  State<EnterWalletPasswordPage> createState() => _EnterWalletPasswordState();
}

class _EnterWalletPasswordState extends State<EnterWalletPasswordPage>
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
              PyrinTitle(text: "Enter your password"),
              PyrinSubtitle(text: "Enter a strong password to secure your wallet."),

              Container(
                margin: const EdgeInsets.only(bottom: 20),
                child: PyrinPasswordTextField(
                  name: "Password",
                  hintText: "Enter your password",
                ),
              ),
            ],
          ),
        );
    }

    onNextClick()
    {

    }
}

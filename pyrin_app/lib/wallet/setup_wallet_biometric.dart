import "package:flutter/material.dart";
import "package:flutter_svg/flutter_svg.dart";
import "package:pyrin_app/core/page.dart";
import "package:pyrin_app/ui.dart";

class SetupWalletBiometricPage extends StatefulWidget
{
  const SetupWalletBiometricPage({super.key});

  @override
  State<SetupWalletBiometricPage> createState() => _SetupWalletBiometricState();
}

class _SetupWalletBiometricState extends State<SetupWalletBiometricPage>
{
    @override
    Widget build(BuildContext context)
    {
        return RoutePage(
          name: "Biometric",
          buttons: [
            PyrinElevatedButton(
              text: "Next",
              onClick: onNextClick,
              wide: true,
            )
          ],
          child: Column(
            children: [
              PyrinTitle(text: "Set up your fingerprint"),
              PyrinSubtitle(text: "Add your fingerprint for a quick access to your wallet."),
              const Expanded(child: const SizedBox()),
              SvgPicture.asset("assets/biometric.svg"),
              const Expanded(child: const SizedBox()),
              PyrinFlatButton(
                text: "Skip for now",
                rightIcon: "arrow-right",
                onClick: (){},
              ),
            ],
          ),
        );
    }

    onNextClick()
    {

    }
}

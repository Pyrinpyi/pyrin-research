import "package:flutter/material.dart";
import "package:pyrin_app/core/page.dart";
import "package:pyrin_app/ui.dart";

class ImportWalletPage extends StatefulWidget
{
  const ImportWalletPage({super.key});

  @override
  State<ImportWalletPage> createState() => _ImportWalletState();
}

class _ImportWalletState extends State<ImportWalletPage>
{
    @override
    Widget build(BuildContext context)
    {
        return RoutePage(
          name: "Import Wallet",
          buttons: [
            PyrinElevatedButton(
              text: "Next",
              onClick: onNextClick,
              wide: true,
            )
          ],
          child: Column(
            children: [
              PyrinTitle(text: "Secret Recovery Phrase"),
              PyrinSubtitle(text: "Enter your Secret Recovery Phrase to restore your wallet."),

              Container(
                margin: const EdgeInsets.only(bottom: 20),
                child: PyrinTextField(
                  name: "Recovery Phrase",
                  maxLines: 5,
                  hintText: "Enter your secret recovery phrase",
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

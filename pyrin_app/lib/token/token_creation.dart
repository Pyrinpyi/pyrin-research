import "package:flutter/material.dart";
import "package:provider/provider.dart";
import "package:pyrin_app/core/page.dart";
import "package:pyrin_app/core/wallet_provider.dart";
import "package:pyrin_app/modals/confirmation.dart";
import "package:pyrin_app/ui.dart";


class TokenCreatePage extends StatefulWidget
{
  const TokenCreatePage({super.key});

  @override
  State<TokenCreatePage> createState() => _TokenDetailsState();
}

class _TokenDetailsState extends State<TokenCreatePage>
{
  @override
  Widget build(BuildContext context)
  {
    return RoutePage(
      name: "Token",
      buttons: [
        PyrinElevatedButton(
          text: "Next",
          onClick: onNextClick,
          wide: true,
        )
      ],
      child: ListView(
        shrinkWrap: true,
        children: [
          PyrinTitle(text: "Token Creation"),
          PyrinSubtitle(text: "Fill the tokens details to submit as new native token on Pyrin Network."),

          Container(
            margin: const EdgeInsets.only(bottom: 20),
            child: PyrinTextField(
              name: "Name",
              hintText: "e.g. Pyrin",
            ),
          ),
          Container(
            margin: const EdgeInsets.only(bottom: 20),
            child: PyrinTextField(
              name: "Symbol",
              hintText: "e.g. PYI",
            ),
          ),
          Container(
            margin: const EdgeInsets.only(bottom: 20),
            child: PyrinTextField(
              name: "Total Supply",
              keyboardType: TextInputType.number,
              controller: TextEditingController(text: 1e8.toStringAsFixed(0)),
            ),
          ),
          Container(
            margin: const EdgeInsets.only(bottom: 20),
            child: PyrinGroup(
              label: "Image",
              child: Container(), // TODO:
            ),
          ),
          Container(
            margin: const EdgeInsets.only(bottom: 20),
            child: PyrinGroup(
              label: "Network",
              child: PyrinTabs(
                onChange: (value) {},
                items:
                [
                    PyrinTabItem(name: "Testnet", value: "testnet"),
                    PyrinTabItem(name: "Mainnet", value: "mainnet"),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  onNextClick()
  {
      final TextStyle textStyle = Theme.of(context).textTheme.bodyMedium!.copyWith(fontWeight: FontWeight.w500);
      final String receiveAddress = Provider.of<WalletProvider>(context, listen: false).receiveAddress;

      showConfirmationModal(
          context: context,
          text: "Token Creation Confirmation",
          fromAddress: receiveAddress,
          toAddress: receiveAddress,
          items:
          [
              ConfirmationItem(name: "Network", child: Text("Testnet", style: textStyle)),
              ConfirmationItem(name: "Estimated Gas Fee", child: Text("0.0001 PYI", style: textStyle)),
              ConfirmationItem(name: "Total", child: Text("1 PYI", style: textStyle)),
          ],
          onConfirm: () async
          {
              await Future.delayed(Duration(seconds: 2));

              // Dismiss the confirmation modal
              Navigator.pop(context);

              transactionConfirmedModal(context);
          }
      );
  }
}

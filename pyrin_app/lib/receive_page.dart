import "dart:math" as math;
import "dart:ui" as ui;
import "package:flutter/material.dart";
import "package:flutter/services.dart";
import "package:flutter_svg/flutter_svg.dart";
import "package:provider/provider.dart";
import "package:pyrin_app/components/addressbook.dart";
import "package:pyrin_app/components/select_token.dart";
import "package:pyrin_app/components/tokens_list.dart";
import "package:pyrin_app/core/token.dart";
import "package:pyrin_app/core/wallet_provider.dart";
import "package:pyrin_app/ui.dart";
import "package:pyrin_app/core/page.dart";
import "package:qr_flutter/qr_flutter.dart";

class ReceivePage extends StatefulWidget
{
    const ReceivePage({super.key});

    @override
    State<ReceivePage> createState() => ReceivePageState();
}

class ReceivePageState extends State<ReceivePage>
{
    String? address;
    Token? token;
    TextEditingController amountController = TextEditingController();

    @override
    Widget build(BuildContext context)
    {
        final double width = MediaQuery.of(context).size.width;

        return RoutePage(
          name: "Receive",
          buttons: [
            Container(
              margin: const EdgeInsets.only(right: 10),
              child: PyrinElevatedButton(
                text: "Share",
                onClick: (){},
                wide: true,
                disabled: false,
                secondary: true,
              ),
            ),
            Container(
              margin: const EdgeInsets.only(left: 10),
              child: PyrinElevatedButton(
                text: "Confirm",
                onClick: () => Navigator.popUntil(context, ModalRoute.withName("/")),
                wide: true,
                disabled: false,
              ),
            ),
          ],
          child: Consumer<WalletProvider>(
                  builder: (context, wallet, child) {
                    return Container(
                      width: double.infinity,
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.center,
                        mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                        children: [
                          QrImageView(
                            data: wallet.receiveAddress,
                            version: QrVersions.auto,
                            eyeStyle: const QrEyeStyle(
                              eyeShape: QrEyeShape.square,
                              color: PyrinColors.WHITE_COLOR,
                            ),
                            dataModuleStyle: const QrDataModuleStyle(
                              dataModuleShape: QrDataModuleShape.square,
                              color: PyrinColors.WHITE_COLOR,
                            ),
                            size: width * 0.5,
                          ),
                          PyrinTextField(
                              controller: TextEditingController(text: wallet.receiveAddress),
                              maxLines: 2,
                              readOnly: true,
                              iconButton: PyrinIconButton(
                                icon: "copy",
                                onClick: () =>
                                    Clipboard.setData(ClipboardData(text: wallet.receiveAddress)),
                              ))
                        ],
                      ),
                    );
                  },
                ),
              );
  }
}

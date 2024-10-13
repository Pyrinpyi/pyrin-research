import "dart:math" as math;
import "dart:ui" as ui;
import "package:flutter/material.dart";
import "package:flutter/services.dart";
import "package:flutter_svg/flutter_svg.dart";
import "package:pyrin_app/components/addressbook.dart";
import "package:pyrin_app/components/select_token.dart";
import "package:pyrin_app/components/tokens_list.dart";
import "package:pyrin_app/core/token.dart";
import "package:pyrin_app/ui.dart";
import "package:pyrin_app/core/page.dart";

class SendPageArguments
{
    final String address;

    SendPageArguments({required this.address});
}

class SendPage extends StatefulWidget
{
    const SendPage({super.key});

    @override
    State<SendPage> createState() => SendPageState();
}

class NumbersKeyboard extends StatelessWidget
{
    final TextEditingController controller;

    NumbersKeyboard({required this.controller});

  @override
    Widget build(BuildContext context)
    {
        final width = MediaQuery.of(context).size.width;

        return GridView.count(
          crossAxisCount: 3,
          crossAxisSpacing: 0,
          mainAxisSpacing: 0,
          childAspectRatio: 1.75,
          padding: EdgeInsets.zero,
          shrinkWrap: true,
          children:
          [
            _buildButton(context, "1"),
            _buildButton(context, "2"),
            _buildButton(context, "3"),
            _buildButton(context, "4"),
            _buildButton(context, "5"),
            _buildButton(context, "6"),
            _buildButton(context, "7"),
            _buildButton(context, "8"),
            _buildButton(context, "9"),
            _buildButton(context, "."),
            _buildButton(context, "0"),
            _buildButton(context, "", child: SizedBox(
              width: 24,
              height: 24,
              child: Stack(
                alignment: Alignment.center,
                children: [
                  SvgPicture.asset("assets/icons/arrow-left.svg", width: 24, height: 24)
                ],
              ),
            )),
          ],
        );
    }

    _buildButton(BuildContext context, String text, {Widget? child})
    {
        return Container(
          child: TextButton(
            onPressed: ()
            {
                if (text == "")
                {
                    final text = controller.text;
                    controller.text = text.substring(0, text.length - 1);
                }
                else
                {
                    controller.text += text;
                }
            },
            style: TextButton.styleFrom(
              padding: EdgeInsets.symmetric(horizontal: 24, vertical: 16),
              backgroundColor: Colors.transparent,
              foregroundColor: PyrinColors.TEXT_COLOR,
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(12),
              ),
              textStyle: TextStyle(
                fontSize: 20,
                fontWeight: FontWeight.w400,
              ),
            ),
            child: child ?? Text(text),
          ),
        );
    }
}

class SendPageState extends State<SendPage>
{
    String? address;
    Token? token;
    TextEditingController addressController = TextEditingController();
    TextEditingController amountController = TextEditingController();

    @override
    Widget build(BuildContext context)
    {
        final bool confirmationStep = addressController.text.isNotEmpty && token != null;

        if (ModalRoute.of(context)!.settings.arguments != null)
        {
            final args = ModalRoute.of(context)!.settings.arguments as SendPageArguments;
            addressController.text = args.address;
        }

        return RoutePage(
          name: "Send",
          buttons: [
            PyrinElevatedButton(
              text: confirmationStep ? "Confirm" : "Next",
              onClick: confirmationStep ? onConfirmClick : onNextClick,
              wide: true,
              disabled: false, // address == null
            )
          ],
          child: !confirmationStep ? _buildRecipientSelection() : _buildAmountSelection(),
        );
    }

    onNextClick()
    {
        pyrinShowModalBottomSheet(
          context: context,
          child: SelectToken(onClick: onTokenClick),
          isScrollControlled: true,
          isDismissible: true,
        );
    }

    onConfirmClick()
    {
        // TODO: Real data
        transactionConfirmedModal(context);
    }

    onTokenClick(Token token)
    {
        setState(()
        {
            this.address = "pyrin"; // TODO:
            this.token = token;
        });

        // Close the bottom sheet
        Navigator.pop(context);
    }

    _buildRecipientSelection()
    {
        return Column(
          children: [
            // TODO: Give an error state when address is invalid
            PyrinTextField(
              controller: addressController,
              name: "Send To",
              hintText: "Enter address",
              iconButton: PyrinIconButton(
                icon: "scan",
                onClick: ()
                {
                  scanAddress(context, (address)
                  {
                    if (mounted)
                    {
                      setState(()
                      {
                          addressController.text = address;
                      });
                    }
                  });
                },
              ),
            ),
            const SizedBox(height: 5),
            PyrinFlatButton(
                text: "Add this to your address book",
                leftIcon: "add",
                onClick: (){},
            ),
            const SizedBox(height: 30),
            AddressBookList(onClick: (item) => {}),
          ],
        );
    }

    _buildAmountSelection()
    {
        return Column(
          children: [
            const SizedBox(height: 24),
            Text("Enter Amount", style: Theme.of(context).textTheme.bodySmall!.copyWith(fontSize: 11, color: PyrinColors.LIGHT_TEXT_COLOR)),
            const SizedBox(height: 12),
            TextField(
              textAlign: TextAlign.center,
              decoration: InputDecoration(
                hintText: "0.0",
                hintStyle: Theme.of(context).textTheme.bodyMedium!.copyWith(fontSize: 36, color: PyrinColors.LIGHT_TEXT_COLOR),
                border: InputBorder.none,
              ),
              style: Theme.of(context).textTheme.bodyMedium!.copyWith(fontSize: 36),
              keyboardType: TextInputType.number,
              inputFormatters: [FilteringTextInputFormatter.allow(RegExp(r"[0-9.]"))],
              controller: amountController,
            ),
            const SizedBox(height: 12),
            PyrinElevatedButton(text: "Max", onClick: ()
            {
                amountController.text = (token!.amount / 1e8).toStringAsFixed(8);
            }),

            const SizedBox(height: 50),
            PyrinGroup(
                label: "Available Balance",
                small: true,
                child: PyrinCard(
                  child: PyrinListViewItem(
                    title: token!.symbol,
                    subtitle: token!.name,
                    leading: Text((token!.amount / 1e8).toStringAsFixed(2)),
                    icon: PyrinCircleIcon(child: TokenIcon(symbol: token!.symbol)),
                    padding: false,
                  ),
                )
            ),

            const SizedBox(height: 20),
            NumbersKeyboard(controller: amountController),
          ],
        );
    }
}

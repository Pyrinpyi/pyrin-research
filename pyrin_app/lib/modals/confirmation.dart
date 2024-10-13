import "package:flutter/material.dart";
import "package:flutter_svg/flutter_svg.dart";
import "package:pyrin_app/components/tokens_list.dart";
import "package:pyrin_app/ui.dart";

class ConfirmationItem
{
    final String name;
    final Widget child;

    ConfirmationItem({required this.name, required this.child});
}

class Confirmation extends StatelessWidget
{
    final String? text;
    final String? fromAddress;
    final String? toAddress;
    final List<ConfirmationItem>? items;
    final Function onConfirm;

    // Max of 2
    final List<String>? tokens;

    Confirmation({
       Key? key,
        this.text,
        this.fromAddress,
        this.toAddress,
        this.items,
        required this.onConfirm,
        this.tokens,
    });

    @override
    Widget build(BuildContext context)
    {
        return Container(
          width: double.infinity,
          height: MediaQuery.of(context).size.height * 0.7 - 65,
          child: Stack(
            children: [
              ListView(
                shrinkWrap: true,
                padding: const EdgeInsets.only(bottom: 100),
                children: [
                  Text(text ?? "Confirmation", style: Theme.of(context).textTheme.bodyMedium!.copyWith(fontWeight: FontWeight.w600)),
                  Container(
                    width: double.infinity,
                    margin: const EdgeInsets.symmetric(vertical: 20),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.center,
                      children: [
                        ...tokens == null ? [
                          PyrinCircleIcon(
                            // darker: true,
                            child: TokenIcon(),
                          )
                        ] : [
                          Row(
                            mainAxisAlignment: MainAxisAlignment.spaceBetween,
                            children: [
                              PyrinCircleIcon(
                                // darker: true,
                                child: TokenIcon(symbol: tokens!.first),
                              ),
                              PyrinCircleIcon(
                                // darker: true,
                                child: SvgPicture.asset("assets/icons/switch.svg"), // TODO:
                              ),
                              PyrinCircleIcon(
                                // darker: true,
                                child: TokenIcon(symbol: tokens!.last),
                              )
                            ],
                          )
                        ],
                        const SizedBox(height: 10),
                        // Text(),
                        // const SizedBox(height: 10),
                      ],
                    ),
                  ),
                  PyrinDivider(),

                  if (fromAddress != null)
                    Row(
                        children: [
                          Text("From:", style: Theme.of(context).textTheme.bodyMedium!.copyWith(fontWeight: FontWeight.w500)),
                          const Expanded(child: const SizedBox()),
                          Address(address: fromAddress!),
                        ]
                    ),
                  if (toAddress != null)
                    Row(
                        children: [
                          Text("To:", style: Theme.of(context).textTheme.bodyMedium!.copyWith(fontWeight: FontWeight.w500)),
                          const Expanded(child: const SizedBox()),
                          Address(address: toAddress!),
                        ]
                    ),
                  if (items != null)
                    ...[
                      ...items!.map((item) => Column(
                        children: [
                          PyrinDivider(),
                          _buildItem(
                              Row(
                                  children: [
                                    Text(item.name, style: Theme.of(context).textTheme.bodyMedium!.copyWith(fontWeight: FontWeight.w500)),
                                    const Expanded(child: const SizedBox()),
                                    item.child,
                                  ]
                              )
                          ),
                        ],
                      )).toList(),
                    ],

                  PyrinDivider(),
                ],
              ),
                Align(
                  alignment: Alignment.bottomCenter,
                  child: PyrinElevatedButton(
                      wide: true,
                      text: "Confirm",
                      onClick: onConfirm
                  ),
                ),
            ],
          ),
        );
    }

    _buildItem(Widget child)
    {
        return Container(
          margin: EdgeInsets.symmetric(vertical: 16),
          child: child,
        );
    }
}

void showConfirmationModal({
    required BuildContext context,
    String? text,
    String? fromAddress,
    String? toAddress,
    List<ConfirmationItem>? items,
    List<String>? tokens,
    required Function onConfirm,
})
{
    pyrinShowModalBottomSheet(
        context: context,
        isScrollControlled: true,
        maxHeightRatio: 0.7,
        child: Confirmation(
            text: text,
            fromAddress: fromAddress,
            toAddress: toAddress,
            items: items,
            onConfirm: onConfirm,
            tokens: tokens,
        )
    );
}

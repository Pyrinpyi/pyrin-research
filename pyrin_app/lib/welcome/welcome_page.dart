import "package:flutter/material.dart";
import "package:pyrin_app/ui.dart";

class WelcomePageStepsIndicator extends StatelessWidget
{
    final int current;
    final int total;

    WelcomePageStepsIndicator({required this.current, required this.total});

    @override
    Widget build(BuildContext context)
    {
        return Container(
          child: Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: List.generate(
              total,
                  (index) => Container(
                    width: 20,
                    height: 4,
                    margin: const EdgeInsets.symmetric(horizontal: 6),
                    decoration: BoxDecoration(
                      color: current == index ? PyrinColors.TEXT_COLOR : PyrinColors.TEXT_COLOR.withOpacity(0.1),
                      borderRadius: BorderRadius.circular(20),
                    ),
              ),
            ),
          ),
        );
    }
}

class WelcomePageItem
{
    final String title;
    final String description;

    WelcomePageItem({required this.title, required this.description});
}

class WelcomePage extends StatefulWidget
{

    @override
    State<WelcomePage> createState() => _WelcomePageState();
}

class _WelcomePageState extends State<WelcomePage>
{
    int currentStep = 0;

    final List<WelcomePageItem> items =
    [
        WelcomePageItem(
            title: "Your Assets, Your Control",
            description: "Manage your digital assets securely with Pyrin Swap. Track, send, and receive with ease."
        ),
        WelcomePageItem(
            title: "Effortless Swapping",
            description: "Seamlessly swap between cryptocurrencies with just a few taps. Fast, easy, and secure."
        ),
        WelcomePageItem(
            title: "Advanced Security",
            description: "Protect your assets with industry-leading security features, including fingerprint authentication."
        ),
        WelcomePageItem(
            title: "The Only Wallet You Need",
            description: "The most user-friendly, non-custodial, blockchain-agnostic wallet for all your digital assets."
        ),
    ];

    @override
    Widget build(BuildContext context)
    {
        final WelcomePageItem item = items[currentStep];

        return Scaffold(
          body: SafeArea(
            child: Column(
              children: [
                Expanded(
                  child: Stack(
                    children: [
                      Container(
                        width: double.infinity,
                        color: Colors.white,
                        child: Image.asset("assets/welcome_banner_${currentStep + 1}.png", fit: BoxFit.cover),
                      ),
                      Positioned(
                        top: 60,
                        right: 20,
                        child: GestureDetector(
                          onTap: () => setState(() => currentStep = 3),
                          child: Text(
                              "Skip",
                              style: Theme.of(context).textTheme.bodyMedium!.copyWith(
                                  fontWeight: FontWeight.w500,
                                  fontSize: 14,
                                  color: PyrinColors.BLACK1_COLOR,
                              )
                          )
                        ),
                      )
                    ],
                  ),
                ),
                const SizedBox(height: 2),
                Expanded(
                  child: Container(
                    padding: EdgeInsets.all(20),
                    child: Column(
                      children: [
                        const SizedBox(height: 20),
                        Text(
                            item.title,
                            textAlign: TextAlign.center,
                            style: Theme.of(context).textTheme.bodyLarge!.copyWith(
                                fontSize: 24,
                                fontWeight: FontWeight.w600,
                            )
                        ),
                        const SizedBox(height: 10),
                        Text(
                            item.description,
                            textAlign: TextAlign.center,
                            style: Theme.of(context).textTheme.bodyLarge!.copyWith(
                                fontSize: 14,
                                fontWeight: FontWeight.w600,
                                color: PyrinColors.TEXT_COLOR.withOpacity(0.4)
                            )
                        ),
                        const Expanded(child: const SizedBox()),
                      ...currentStep <= 2 ? [
                          WelcomePageStepsIndicator(current: currentStep, total: 3),
                          const SizedBox(height: 40),
                          PyrinPrimaryCircleButton(
                            icon: "arrow-right-short",
                            iconSize: 20,
                            onClick: _nextStep,
                          )
                      ] : [
                          PyrinFlatButton(
                              text: "I already have a wallet",
                              onClick: ()
                              {
                                  Navigator.pushReplacementNamed(context, "/wallet/import");
                              }
                          ),
                          PyrinElevatedButton(
                              wide: true,
                              text: "Create new wallet",
                              onClick: ()
                              {
                                  Navigator.pushReplacementNamed(context, "/wallet/create");
                              }
                          )
                      ],
                      ],
                    ),
                  ),
                ),
              ],
            ),
          ),
        );
    }

    void _nextStep()
    {
        setState(()
        {
            currentStep++;
        });
    }
}
